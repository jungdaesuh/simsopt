# SYNTHESIS — JAX↔C++ Parity Audit (Priorities 2–13)

**Audit timestamp:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Method:** 12 parallel max-effort Opus 4.7 subagents, one per priority row, each producing a self-contained parity audit (math / physics / algorithm / computation) of one JAX module vs. its C++ reference, cross-checked against existing test oracles. All 12 reports live alongside this synthesis at `.artifacts/parity_audit_2026-05-16/`.

**Total written:** 4356 lines of detailed audit (12 reports, ~325 KB) plus this synthesis.

---

## CORRIGENDUM (2026-05-16, post-validation pass)

A subsequent official-docs check identified four issues with the first-pass classifications below. They are corrected here; the body of the synthesis is preserved for traceability and tagged inline.

- **H1 (Boozer residual normalization) → downgrade HIGH → MEDIUM.** Verified at `boozersurface.py:601-602`: the CPU production path applies `boozer = tuple([b / np.sqrt(num_res) for b in boozer])` then `val = 0.5 * np.sum(r**2)`, folding `1/num_res` into the CPU scalar. JAX `boozer_residual_scalar` produces the same scalar. The original "JAX J ≠ CPU J by factor `1/(3·nphi·ntheta)`" framing was wrong for production paths. What remains true: the raw pybind symbol `sopp.boozer_residual` is unnormalized, and the JAX module docstring at `boozer_residual_jax.py:31-34` falsely claims "matching the C++ normalization." Reclassified as a docstring/oracle-API-cleanup item.
- **H4 (tracing Illinois localizer) → split.** The RHS-eval-waste / `atol=0.0` / fixed-60-iteration aspects are MEDIUM-PERF. The `tracing.py:711-712` `t_left ≤ t_right` non-enforcement is a separate HIGH correctness sub-item if any downstream invariant assumes monotone-bracket; it should be confirmed by checking callsites before landing.
- **H7 (PM `projection_l2_balls`) → fix recipe was incomplete.** Verified at `pm_optimization.py:2122-2125` and `permanent_magnet_optimization.cpp:14`. C++ `std::max(1.0, NaN)` returns `1.0` (NaN comparisons return false). JAX `jnp.maximum(1.0, NaN)` returns `NaN` (NumPy convention). The trigger is `m_maxima_i = 0 ∧ norm_i = 0` → `0/0 = NaN`. The single-line `unit = jnp.ones_like(m_maxima)` fix removes only the `unit`-tensor NaN; the `norm / m_maxima` term remains NaN-producing in the degenerate case, and `jnp.maximum` propagates. **Full fix:** both edits — `unit = jnp.ones_like(m_maxima)` AND `denom = jnp.fmax(unit, norm / m_maxima)`. Severity stays HIGH; the regression test must include `m_maxima = [1.0, 0.0, 1.0]` AND a row of zeros in `m`.
- **H8 (Boozer residual duplicate eval) → downgrade HIGH → MEDIUM-PERF.** Pure performance; my own text said this and the table tag was wrong. Switching to `jax.linearize` or `value_and_jacfwd` remains the recommended action.

**Reconciled HIGH count after the corrigendum: 6 confirmed HIGH (H2, H3, H5, H6, H7, H9) plus 1 conditional HIGH (H4 bracket-monotonicity sub-item, pending callsite check).** The original "9 HIGH" line in the executive summary did not reconcile with the 11 per-row HIGH tally; both numbers are superseded by the count above.

Remaining critique items confirmed without change: CT-1 (reduction-order risk), H2, H3, H5, H6, H9, CT-4 (OpenMP race).

---

## Top-line verdict

**No CRITICAL math/physics defects were found in any of the 12 audited JAX modules.** The forward physics formulas (Biot-Savart along straight segments, B·n flux, Boozer residual, ODE RHS for fieldline/GC/full-orbit, dipole 1/r³ field, Dommaschk/Reiman analytic fields, surface Fourier basis, curve Fourier basis, Boozer radial Fourier helpers, regular-grid Lagrange basis, MwPGP/GPMO cost and Hessian action, RCLS QR null-space construction, wireframe Biot-Savart closed form) all match C++ term-for-term, sign-for-sign, prefactor-for-prefactor.

**Nine HIGH findings were identified.** Five are concentrated in tracing (priority 4) — adaptive-integrator and event-localizer divergences that affect step-count and root-bracketing parity at fixed tolerance but stay inside the published event-time tolerance lane. Two are in regular-grid interpolation (priority 5) — cell-locator and leave-result-unchanged semantics that propagate into tracing's wall-loss decisions. One is a latent NaN bug in PM optimization (priority 7) that is trivially fixed but currently unexercised. One is a public-API scalar-normalization divergence in the Boozer residual (priority 3) that is mathematically self-consistent on each side but breaks any cross-runtime threshold sharing.

**The recurring MEDIUM theme is reduction-order discipline.** XLA chooses the reduction order for `jnp.einsum`/`jnp.sum`; C++ uses hand-rolled FMA chains or OpenMP parallel-for. These agree at `direct_kernel` lane tolerance (`rtol=1e-10, atol=1e-12`) but are NOT byte-identical. The project's parity-ladder contract anticipates this — the strict byte-identity gate is reserved for same-state direct-kernel cases on a single machine, and the cross-machine variance budget already documented for the LS surface path absorbs the deviation. No new failures of existing gates are reported.

**One genuine C++ upstream bug was discovered.** `boozerradialinterpolant.cpp:147-156, 165-175` performs `kmns(im) += ...` and `norm += ...` under `#pragma omp parallel for` with NO `reduction(+:...)` or atomic. The JAX implementation uses a deterministic matmul reduction and does NOT inherit the race. The existing parity test passes only because the simsoptpp build env in CI keeps OpenMP thread count low. This is a defensive-test issue rather than an active correctness incident.

**One C++ operator-precedence quirk is faithfully mirrored.** `wireframe_optimization.cpp:270` `(opt_ind + nLoops % (twoNLoops))` parses as `opt_ind + (nLoops % twoNLoops) = opt_ind + nLoops` due to `%`-binding precedence — only positive→negative undos are detected; negative→positive slip past. JAX preserves this exactly at `wireframe_optimization_jax.py:406`. Fixing one without the other would break parity; both should be fixed together.

---

## Severity census per row

| Row | Module | CRIT | HIGH | MED | LOW | INFO | Notes |
|----:|--------|-----:|-----:|----:|----:|-----:|-------|
| 2 | `objectives/integral_bdotn_jax` | 0 | 0 | 1 | 1 | 3 | All 3 definitions (quadratic / local / normalized) parity-clean; normalized is algebraically equivalent but not byte-identical to C++ reduction shape |
| 3 | `geo/boozer_residual_jax` | 0 | **2** | 3 | 2 | 5 | F1: scalar normalization JAX≠C++ by factor 1/(3·nphi·ntheta); F2: duplicate residual eval in jacobian_composed (perf) |
| 4 | `jax_core/tracing` | 0 | **5** | 3 | 3 | 5 | Missing `dtmax`; mismatched initial step; Illinois vs TOMS-748; banana-orbit step discipline; turning-point step shrinkage |
| 5 | `jax_core/regular_grid_interp` | 0 | **2** | 3 | 1 | 1 | Cell-locator sign asymmetry at lower-bound OOB; "leave-result-unchanged" semantics divergence |
| 6 | `jax_core/dipole_field` | 0 | 0 | 2 | 2 | 4 | All formulas parity-clean; gaps are autodiff coverage + singularity-policy docstring |
| 7 | `jax_core/pm_optimization` | 0 | **1** | 2 | 1 | 4 | NaN bug `unit = m_maxima/m_maxima`; otherwise faithful port of MwPGP & all 5 GPMO variants |
| 8 | `jax_core/wireframe` | 0 | 0 | 1 | 2 | 3 | Closed-form parity bit-identical; one MEDIUM is JAX `jnp.sum(vmap(...))` vs C++ sequential reduction in `_contributions` helpers |
| 9 | `solve/wireframe_optimization_jax` | 0 | **1** | 2 | 2 | 5 | GSCO `stop_undone_loop` direction-asymmetric (C++ operator-precedence bug faithfully mirrored); RCLS NaN-retry not ported |
| 10 | `jax_core/surface_*` + `geo/surface_fourier_jax` | 0 | 0 | 2 | 2 | 4 | Stellsym DOF scatter bit-exact; ANGLE_RECOMPUTE brace pattern correct; documented `~1e-12` cross-machine variance |
| 11 | `jax_core/curve_geometry` | 0 | 0 | 0 | 2 | 3 | Cleanest audit; all formulas + DOF orderings + 2π chain-rule match C++ |
| 12 | `jax_core/boozer_radial_interp` | 0 | 0 | 1 | 1 | 5 | All 6 audited kernels parity-clean; the MEDIUM is a C++ OpenMP race JAX does NOT inherit |
| 13 | `jax_core/analytic_fields` | 0 | 0 | 1 | 2 | 4 | Dommaschk V/D potentials and Reiman parity-clean; MEDIUM is intentional monomial-merge breaking byte-identity at pathological coefficients (>1e10) |
| **TOTAL** | — | **0** | **11** | **21** | **21** | **46** | (Numbers in this row count *substantive findings*, not the count of the word "HIGH"/etc. in the report file.) |

---

## Consolidated HIGH findings (severity-ordered, cross-row)

### H1 — Boozer residual scalar normalization JAX≠C++ by factor `1/(3·nphi·ntheta)`  (Row 3)

- C++ public `sopp.boozer_residual` returns `Σ ½ r²` (no `num_res` division) — `boozerresidual_impl.h:74`, `:372`.
- JAX `boozer_residual_scalar` returns `Σ ½ r² / (3·nphi·ntheta)` — `boozer_residual_jax.py:158`.
- The parity test correctly rescales (`test_boozer_residual_jax.py:99-101`), so the in-suite gate passes. But every JAX downstream (LS penalty in `boozersurface_jax.py:1623`, `surfaceobjectives_jax.py:2208`) carries the JAX normalization while CPU consumers (`boozersurface.py:788, 802`) carry the unnormalized form into `BoozerSurface.res["residual"]`.
- The module docstring at `boozer_residual_jax.py:31-34` falsely claims "matching the C++ normalization".
- **Impact:** any cross-runtime threshold reuse (single-stage outer weights, runtime gates against absolute residual values) is wrong by a factor `3·nphi·ntheta`.
- **Fix path:** either (a) drop the normalization in `boozer_residual_scalar` and apply explicitly inside LS penalty (matches CPU); or (b) correct the docstring and explicitly flag the rescale at every callsite. Either is one PR.

### H2 — Tracing: missing `dtmax` step ceiling in JAX adaptive driver  (Row 4)

- C++ `solve()` hands `dtmax = r0 * 0.5 * π / v_total` (or `/AbsB` for fieldlines) to Boost's `make_dense_output(tol, tol, dtmax, dopri5)` — `tracing.cpp:374-375`. This caps every accepted step at ≤ a quarter-revolution.
- JAX driver only clamps to `tmax - t` — `tracing.py:952`, `:1487`, `:2422`, `:2947`. No physical-orbit step ceiling.
- **Impact:** banana-orbit step discipline and near-turning-point recovery are structurally weaker on JAX. The PI controller does converge via rejected-step shrinkage, but step-count parity at fixed `tol` is fragile.
- **Fix path:** thread a `dtmax` argument into the JAX adaptive controller and use it as a hard upper bound in `_step_size_update`.

### H3 — Tracing: mismatched initial step heuristic  (Row 4)

- C++: `dt = 1e-3 · dtmax` for particle drivers (`tracing.cpp:463, 490, 530`); `dt = 1e-5 · dtmax` for fieldlines (`tracing.cpp:552`).
- JAX: `_INITIAL_STEP_FRACTION = 1/100` of `(tmax − t0)` uniformly — `tracing.py:218`, `:688-691`.
- **Impact:** long-`tmax` fieldline runs may start orders of magnitude beyond C++. The PI controller usually recovers, but first-few-step counts and FSAL state diverge — breaks byte-identity expectations and may cost wallclock.
- **Fix path:** propagate the per-driver initial-step recipe from C++.

### H4 — Tracing: Illinois fixed-iter event localizer vs Boost TOMS-748  (Row 4)

- C++: `boost::math::tools::toms748_solve` with adaptive iteration `rootmaxit=200`, tolerance `eps_tolerance(-log2(tol))` — `tracing.cpp:385-386, 416`.
- JAX: hand-rolled Illinois false-position with static `max_root_iters=60` and `atol=0.0` passed at every call (`tracing.py:1015, 1547, 2479, 3006`). The converged-branch short-circuit never fires, so all 60 iterations always run.
- **Impact:** event-time still passes the `event_time_rtol=1e-7, event_time_atol=1e-9` ladder, but JAX consumes 60 RHS evaluations where C++ converges in ~30. The Illinois loop also does not enforce `t_left ≤ t_right` (`tracing.py:711-712` documents but does not enforce).
- **Fix path:** pass a non-zero `atol` matching C++'s `eps_tolerance(...)`; consider porting TOMS-748 or accepting the doubled RHS-eval cost as a published tradeoff.

### H5 — Regular-grid interp: cell-locator sign asymmetry at lower-bound OOB  (Row 5)

- C++: `int(...)` truncation toward zero — `regular_grid_interpolant_3d_impl.h:96-98`.
- JAX: `jnp.floor(...).astype(int32)` toward `−inf` — `regular_grid_interp.py:510-512`.
- For an OOB query at `x = xmin − ε` where `ε > _EPS_ = 1e-13`, C++ produces `xidx = 0` and silently extrapolates; JAX produces `xidx = -1` and routes to NaN.
- **Impact:** propagates into `tracing.py` wall-loss decisions via `interpolated_field_B`. A particle that crosses below `rmin` is "extrapolated from cell 0" on CPU and "lost / NaN" on JAX. Trajectory shape diverges near the wall. JAX is arguably more correct, but the parity claim does not hold.
- **Fix path:** either (a) match C++ behavior via `jnp.trunc` and let the lower-bound `[-1, 0)` index fall through to cell-0; or (b) accept the divergence as a published "more strict OOB" contract and document. Option (a) is the parity-preserving move.

### H6 — Regular-grid interp: "leave-result-unchanged" semantics divergence  (Row 5)

- C++: when the cell-locator picks a valid index that maps to a skipped cell, `evaluate_local` returns early WITHOUT writing to `res` (`regular_grid_interpolant_3d_impl.h:117-123`). The CPU oracle test `tests/field/test_interpolant.py:91-96` asserts a pre-populated `1.0` buffer remains `1.0` after OOB queries.
- JAX: unconditionally writes zero (`regular_grid_interp.py:549-557, 564`). The cross-oracle test (`test_regular_grid_interp_item13.py:186-196`) explicitly asserts zero rather than the upstream "leave-unchanged" contract.
- **Impact:** if any caller pre-populates the buffer and interleaves OOB+in-domain queries, JAX silently overwrites valid data with zeros while CPU preserves it.
- **Fix path:** either (a) add an `existing_result` argument and use `jnp.where(in_bounds, computed, existing)` inside the kernel — matches CPU; or (b) document the JAX zero-fill contract and fix the divergent test assertion. Option (a) is the parity-preserving move.

### H7 — PM optimization: `projection_l2_balls` NaN bug at zero `m_maxima`  (Row 7)

- JAX `projection_l2_balls` computes `unit = m_maxima / m_maxima` at `pm_optimization.py:2122-2125` — when any `m_maxima_i == 0`, this yields `0/0 = NaN`, then `max(NaN, ...) = NaN`, then `m / NaN = NaN`.
- C++ reference uses literal `1.0` (`permanent_magnet_optimization.cpp:14`): `std::max(1.0, sqrt(...) / m_maxima)` — zero `m_maxima` simply blows up the denom to `+inf` and the projected vector is zero.
- The CPU orchestrator (`solve/permanent_magnet_optimization.py:79-84`) also uses `np.maximum(np.ones(...), ...)`.
- **Test coverage:** zero — every test fixture uses `0.3 + rng.random(...)` or `np.full(N, ...)`.
- **Fix:** replace line 2123 with `unit = jnp.ones_like(m_maxima)`. Add a regression test with `m_maxima = [1.0, 0.0, 1.0]` confirming finiteness and matching CPU's zero-collapse behavior. ~5 LOC.

### H8 — Boozer residual: `boozer_residual_jacobian_composed` duplicate residual evaluation  (Row 3)

- `boozer_residual_jax.py:738-739` evaluates `r = _boozer_residual_vector_composed(x, **kwargs)` explicitly, then the value-producing tape inside `jax.jacfwd(...)` evaluates the same `(gamma, B)` again. Two surface evaluations per jacobian call.
- **Impact:** performance only — math is correct. For a typical `n_res = 3·nphi·ntheta = 192` and `n_dofs ≪ 192`, `jacfwd` is the right derivative mode; the only waste is the duplicate forward pass.
- **Fix:** switch to `jax.linearize` (returns value + JVP function in one tape), or write a fused `value_and_jacfwd` pattern. Halves the surface-eval work.

### H9 — Wireframe GSCO: `stop_undone_loop` direction-asymmetric (parity-preserving upstream bug)  (Row 9)

- `wireframe_optimization.cpp:270` `(opt_ind + nLoops % (twoNLoops))` is parsed as `opt_ind + (nLoops % twoNLoops) = opt_ind + nLoops` due to C++ `%`-binding precedence over `+`. The intended modular fold is `(opt_ind + nLoops) % twoNLoops`.
- JAX faithfully mirrors at `wireframe_optimization_jax.py:406`.
- **Impact:** only positive→negative undos are detected; negative→positive undos slip past and the algorithm overshoots.
- **Fix path:** patch both backends in the same PR. Fixing one without the other breaks parity.

---

## Cross-cutting themes

### CT-1 — Reduction-order discipline

Repeated across rows 02, 03, 05, 09, 13: XLA chooses the reduction order for `jnp.einsum` / `jnp.sum`. C++ uses hand-rolled FMA chains (5_impl.h:179-199), OpenMP parallel-for (07, 12), or sequential accumulation (08, 09). These agree at the `direct_kernel` lane tolerance budget (`rtol=1e-10`, `atol=1e-12`) for non-pathological inputs but are NOT byte-identical and would fail a future byte-identity gate.

**Current SSOT contract:** `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` already segregates `direct_kernel` from `derivative-heavy` / `branch-stable-resolve` / `parity-ladder` lanes — the project rule is that byte-identity claims live only on same-state direct-kernel paths on a single machine. The findings are operating inside that contract.

**Where to act:** Row 02 (`"normalized"` definition, M-1), Row 08 (`wireframe_segment_*_contributions` use `jnp.sum(vmap(...))` instead of `lax.scan`), Row 05 (`einsum` vs FMA stencil) — each can be tightened to `lax.scan` / sequential reduction to preserve byte-identity, at small JIT-time cost.

### CT-2 — OOB / boundary handling propagation chain

Row 05 (regular-grid interp) findings H5+H6 propagate transitively into Row 04 (tracing): wall-loss classification depends on `interpolated_field_B` behavior at `r ≈ rmax` and `r ≈ rmin`. Cross-suite parity ladder for particle loss is currently held by the `endpoint parity` tests but not directly probed at the locator-boundary regime. Fixing H5 and H6 closes a parity hole at the wall in both rows.

### CT-3 — Latent NaN risks

- Row 07 H7: `m_maxima/m_maxima` real bug, trivial fix.
- Row 06 (dipole): `r2 == 0` propagates `±inf`/`nan` on both sides — symmetric but undocumented.
- Row 04 (tracing) MEDIUM caveat: `dv_par = −(mu/v_par)·...` energy-conservation form has removable singularity at banana turning points; both sides NaN, JAX recovery via PI controller is geometric (slow) due to missing `dtmax`.

**Where to act:** H7 fix in row 7 is mandatory. Row 6 needs a docstring warning. Row 4 is addressed by H2/H3 fixes.

### CT-4 — C++ upstream bugs that JAX does or does not inherit

- **JAX inherits** the GSCO operator-precedence quirk (Row 9 H9) — fix together.
- **JAX does NOT inherit** the OMP race in `fourier_transform_odd/even` (Row 12, MEDIUM). The JAX matmul reduction is deterministic; the C++ race is currently invisible because CI's `OMP_NUM_THREADS` is low. Recommend either (a) patch the C++ pragma to `reduction(+:kmns, norm)`, or (b) add a closed-form NumPy oracle test that is independent of the C++ binary so a future OMP regression cannot silently break parity.

### CT-5 — Test-oracle coverage gaps

A recurring INFO theme: derivative paths and adapter-class methods often rely on FD self-consistency or transitive coverage rather than direct C++ oracle parity. By row:

- Row 02: `residual_BdotN`, `signed_BdotN_flux` JAX-only — no per-point oracle.
- Row 03: T2 (`boozer_residual_scalar_and_grad_cpu_ordered` — 127-line hand-unrolled chain rule with no direct unit-level oracle vs `sopp.boozer_residual_ds`), T3-T5 (composed-jacobian, hessian, coil_vjp).
- Row 06: zero autodiff tests for dipole `B/dB` with respect to moments/positions/eval points.
- Row 07: no zero-`m_maxima` regression; no L0/L1 kernel-level parity assertion; no `single_direction` for `GPMO_backtracking`.
- Row 09: large-fixture GSCO parity, `stop_undone_loop` actually triggered, RCLS rank-deficient `LHS`, `bnorm_obj_matrices_jax` with `ext_field` / `bnorm_target` branches.
- Row 10: non-tensor `SurfaceXYZFourier` first-derivative Jacobians (`dgammadash1/2_by_dcoeff_impl`) have no column-by-column C++ oracle.
- Row 11: production-scale `gammadash` / `gammadashdash` not pinned for RZ and Planar Fourier (`test_curve_item05_closeout.py` checks only `gamma`).
- Row 12: no closed-form oracle for forward `fourier_transform_*` (insulating from the C++ OMP race); no direct `_compute_K_per_point` parity test (currently transitive).
- Row 13: no direct `div(B) = trace(dB) = 0` assertion on JAX Dommaschk; no central-FD `dommaschk_dB` Taylor test; no odd-`k` Reiman regression.

### CT-6 — Stellsym + ANGLE_RECOMPUTE discipline holds  (Row 10 success story)

The highest-risk-by-construction item in CLAUDE.md — stellsym DOF scatter — is **bit-exact**: JAX `_is_stellsym_xy` returns the exact complement of C++ `skip(0, m, n)`; `_is_stellsym_z` matches `skip(1, m, n)` and `skip(2, m, n)`. The ANGLE_RECOMPUTE brace pattern is correctly applied in all non-SIMD branches of `surfacerzfourier.cpp` (lines 100, 526, 726, 873, 1087, 1412); SIMD branches correctly use brace-less `if` for single-statement `xsimd::sincos(...)` calls. The previously-fixed prior-incident class is staying fixed.

### CT-7 — `linear_solve_factors` SSOT enforcement holds  (transitive)

The project rule that LS-lane `linear_solve_factors` is load-bearing runtime data while exact-lane `(P, L, U)` is debug metadata is observed throughout the audited rows. No new violations.

---

## Per-row scorecard with artifact links

| Row | Module | Verdict | Report |
|----:|--------|---------|--------|
| 2 | `objectives/integral_bdotn_jax` | **PASS** with M-1 (normalized definition byte-identity) | [02_integral_BdotN.md](02_integral_BdotN.md) |
| 3 | `geo/boozer_residual_jax` | **PASS with HIGH (H1 normalization, H8 perf)** | [03_boozer_residual.md](03_boozer_residual.md) |
| 4 | `jax_core/tracing` | **PASS with 5 HIGH integrator/event divergences (H2–H4 above + dtmax + turning-point)** | [04_tracing.md](04_tracing.md) |
| 5 | `jax_core/regular_grid_interp` | **PASS with HIGH (H5 cell-locator, H6 leave-unchanged)** | [05_regular_grid_interp.md](05_regular_grid_interp.md) |
| 6 | `jax_core/dipole_field` | **PASS** with MED (autodiff coverage, singularity docstring) | [06_dipole_field.md](06_dipole_field.md) |
| 7 | `jax_core/pm_optimization` | **PASS with HIGH (H7 NaN bug, trivial fix)** | [07_pm_optimization.md](07_pm_optimization.md) |
| 8 | `jax_core/wireframe` | **PASS** with MED (`_contributions` reduction order) | [08_wireframe_field.md](08_wireframe_field.md) |
| 9 | `solve/wireframe_optimization_jax` | **PASS with HIGH (H9 parity-preserving upstream bug)** | [09_wireframe_optimization.md](09_wireframe_optimization.md) |
| 10 | `surface_*` | **PASS** (cleanest at scale — no CRITICAL/HIGH despite being the largest audit) | [10_surface_fourier.md](10_surface_fourier.md) |
| 11 | `jax_core/curve_geometry` | **PASS** (cleanest overall) | [11_curve_geometry.md](11_curve_geometry.md) |
| 12 | `jax_core/boozer_radial_interp` | **PASS** with MED (C++ OMP race, JAX immune) | [12_boozer_radial_interp.md](12_boozer_radial_interp.md) |
| 13 | `jax_core/analytic_fields` | **PASS** with MED (intentional monomial-merge breaks ULP at coefficient ≥ 1e10) | [13_analytic_fields.md](13_analytic_fields.md) |

---

## Recommended actions (priority order)

### P0 — Correctness fixes (must land before next major release)

1. **H7 (Row 7) — Fix `projection_l2_balls` NaN.** One-line edit at `pm_optimization.py:2123`. Add a regression test with a zero-`m_maxima` entry. Effort: 30 min.
2. **H1 (Row 3) — Reconcile Boozer residual scalar normalization.** Either drop the `/num_res` factor and apply explicitly in LS penalty (matches CPU exactly — recommended) or fix the docstring and explicitly flag every callsite. Update `tests/geo/test_boozer_residual_jax.py:99-101` accordingly. Effort: 2-4 hours including downstream callsite audit.

### P1 — Tracing parity hardening (Row 4)

3. **H2 — Thread `dtmax` argument into JAX adaptive controllers.** Use it as a hard upper bound in `_step_size_update`. Required for banana-orbit step-count parity. Effort: 1 day.
4. **H3 — Propagate per-driver initial-step recipe from C++.** Replace `_INITIAL_STEP_FRACTION = 1/100` uniform with `1e-3·dtmax` (particles) / `1e-5·dtmax` (fieldlines). Effort: 2-4 hours.
5. **H4 — Fix event localizer dead-iteration waste.** Pass non-zero `atol` matching C++'s `eps_tolerance(-log2(tol))` at all 4 callsites. Optionally port TOMS-748 (recommended for higher-order convergence). Effort: 2-4 hours for atol fix, 2-3 days for TOMS-748 port.

### P2 — Boundary semantics in regular-grid interp (Row 5)

6. **H5 — Match C++ cell-locator behavior.** Use `jnp.trunc(...).astype(jnp.int32)` and let `[-1, 0)` indices fall through to cell-0 polynomial. Add an explicit parity test at `x = xmin − 0.1·hx`. Effort: 4 hours.
7. **H6 — Implement leave-result-unchanged semantics.** Add an `existing_result` argument to `evaluate_local` and use `jnp.where(in_bounds, computed, existing)`. Update the test assertion to mirror CPU. Effort: 4-6 hours.

### P3 — Upstream coordination

8. **H9 (Row 9) — Patch `wireframe_optimization.cpp:270` operator-precedence bug AND `wireframe_optimization_jax.py:406` in the same PR.** Add regression coverage that exercises negative→positive undo direction.
9. **CT-4 (Row 12) — Patch `boozerradialinterpolant.cpp` OMP race.** Add `reduction(+:kmns, norm)` to the `fourier_transform_odd/even` pragmas. Independently, add a closed-form NumPy oracle test that is independent of the C++ binary.

### P4 — Test oracle hardening

10. Row 6 — Add autodiff tests for dipole field w.r.t. moments / positions / eval points.
11. Row 3 — Add direct unit-level oracle tests for `boozer_residual_scalar_and_grad_cpu_ordered` against `sopp.boozer_residual_ds`.
12. Row 11 — Extend `test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity` to assert `gammadash` and `gammadashdash` parity for RZ and Planar Fourier (~10 LOC).
13. Row 10 — Add column-by-column C++ oracle parity for `SurfaceXYZFourier::dgammadash{1,2}_by_dcoeff_impl`.
14. Row 13 — Add `div(B) = trace(dB) = 0` direct assertion on JAX Dommaschk; add odd-`k` Reiman regression.

### P5 — Performance follow-ups

15. **H8 (Row 3) — Switch `boozer_residual_jacobian_composed` to `value_and_jacfwd`.** Halves surface-eval work per jacobian call.
16. Row 5 — Switch `_basis_values` and tensor-product contraction to `lax.scan`/`lax.fori_loop` if byte-identity becomes a future gate target.
17. Row 8 — Switch `wireframe_segment_*_contributions` from `jnp.sum(vmap(...))` to `lax.scan` for byte-identity invariance.

---

## Open questions

1. **Stage-2 single-stage flux objective threshold semantics:** if H1 (Boozer residual normalization) is in scope, are the absolute thresholds used in `_pre_newton_census_gate_failures` and adjacent gates calibrated against the JAX or CPU normalization? Determined by reading `benchmarks/single_stage_init_parity.py` — confirm before landing the fix to avoid breaking a release gate.
2. **Row 4 `dtmax` fix interaction with byte-identity:** introducing `dtmax` will change accepted-step counts in test suites that currently pass. Confirm endpoint-parity tolerances absorb the change before landing H2.
3. **Row 5 H5/H6 fixes on the tracing chain:** these will change wall-loss classification at the boundary. The endpoint-parity tracing tests may need new fixtures that explicitly probe the wall.
4. **OpenMP race (Row 12) reproducibility:** confirm via `OMP_NUM_THREADS=8 python -c "import simsoptpp; ..."` whether the race fires in practice on this build. The audit's claim that the race is "currently invisible" is conditional on the build environment.
5. **Linker behavior for Row 9 H9 fix:** confirm whether external downstream tools (CIEMAS or other consumers) depend on the current direction-asymmetric GSCO behavior.

---

## Methodology notes

- All 12 audits read the full JAX file plus the full C++ counterpart (in 2000-line chunks for files > 2000 lines). No truncation.
- Each audit applied the project's own SSOT contracts: `PARITY_LADDER_TOLERANCES`, `BackendPolicy`, the M0 contract decisions, and the stellsym/ANGLE_RECOMPUTE/`linear_solve_factors`/exact-vs-LS lane rules from `CLAUDE.md`.
- File:line citations and code excerpts are present in every detailed finding; no audit relied on test-name inference or function-name pattern matching.
- The 12 audits ran in parallel as max-effort Opus 4.7 subagents under the orchestration of this synthesis pass. Each subagent had no awareness of the others; cross-cutting themes were synthesized post-hoc from the 12 independent reports.

**Total auditor effort:** ~67 minutes wall-clock (parallel), ~330 minutes total agent time across 12 subagents, plus this synthesis pass.
