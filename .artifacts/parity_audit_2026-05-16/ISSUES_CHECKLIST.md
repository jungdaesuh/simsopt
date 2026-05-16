# JAX↔C++ Parity Audit — Consolidated Issues Checklist

**Source:** First-pass audit (12 rows) + corrigendum + deeper-pass audit (12 rows) + checklist-validation corrigendum (2026-05-16).
**Date:** 2026-05-16
**Branch:** `gpu-purity-stage2-20260405`
**Total items:** 167 unchecked + 2 pre-checked meta entries = 169 total, across 11 sections (10 priority tiers P0-P8 + P5+ plus a "Pass-4 additions" recovery section). 161 actionable + 6 out-of-scope + 2 meta-resolved. Verified by `grep -c "^- \[ \]"`. The Pass-4 additions section recovers ~67 items present in the per-row reports that were consolidated or dropped during the original checklist build; each item is tagged with its natural priority tier in `[Px]` brackets.

Legend: severity prefix in description; `(LIVE)` means subagent reproduced the issue against the running env; effort estimates in parentheses.

---

## CORRIGENDUM (2026-05-16, post-validation pass)

A two-stage external validation against (a) live source and (b) official JAX/NVIDIA/SIMSOPT docs identified gaps in the first cut of this checklist. Corrections applied:

- **Counts corrected (historical pass-chain):** original header said "89 issues across 7 priority tiers"; Pass-3 corrected to 101 unchecked items (95 actionable + 6 OOS) across 10 priority tiers; Pass-4 expanded to 169 unchecked items / 163 actionable after recovering ~67 row-report items. **Current state (Pass-5, after meta-checking 2 reconciliation entries): 167 unchecked + 2 checked meta = 169 total; 161 actionable + 6 out-of-scope + 2 meta-resolved** — see top of file. Summary table at bottom reflects current state.
- **F1 expanded** from JAX core only to cover CPU `projection_L2_balls` (`permanent_magnet_optimization.py:82`) AND CPU+JAX `prox_l1` (`:63` and `permanent_magnet_optimization_jax.py:320`). Both divide by `m_maxima` without zero-guard. Live-verified: `m_maxima=[1,0,1]` with zero `m` row produces NaN in all three lanes.
- **DH17 added** (P3 dipole non-cartesian on-axis silent C++↔JAX divergence — was in deeper synthesis but missed in checklist).
- **DH25 added** (P3 wireframe `np.int32` narrowing at `wireframefield_jax.py:26` — was in deeper synthesis but missed in checklist).
- **CPU validation-contract drift added** to P7 (JAX `dipole_field.py:439` raises on shape mismatch; C++ `dipole_field.cpp:310` silently reads OOB).
- **Reachability finding added** to P0 F1 scope: `cell_vol = 0` for on-axis cylindrical cells at `permanent_magnet_grid.py:382` makes `m_maxima = 0` reachable in production; the zero-`m_maxima` NaN bug is NOT a corner case.
- **F2 reclassified** from P0 to P7: it is a docstring correction, not a correctness blocker. Production scalar paths are already equivalent per the first synthesis corrigendum.
- **Deeper-pass synthesis discrepancy noted:** the deeper synthesis text says "19 new HIGH" but its census table totals 26. The 26 figure is authoritative (per-row tally).
- **F-DH8 scope (superseded by Pass-4):** initial Pass-3 narrowing claimed the missing back-fill only matters at budget exhaustion. Pass-4 re-broadened this — source report says normal-exit `loss_ctr` can be structurally biased on JAX even for cleanly-stopped trajectories. **F-DH8 stands at its original broad scope** (every non-terminated trajectory). See R04-A1 in Pass-4 additions for the regression test that exposes this.

Official-doc check confirmed:
- **F1**: JAX docs confirm `jnp.maximum` propagates NaN; `jnp.fmax` returns finite operand. Live-verified.
- **CT-D1 / P4**: JAX FAQ confirms `where`/NaN-gradient hazard.
- **F-DH1**: JAX `lstsq` docs say `rcond=None` uses internal cutoff; does NOT prove SciPy parity. F-DH1 remains valid.
- **P8 / F20**: NVIDIA CUDA docs confirm reduction/order/FMA flexibility. Reaffirms byte-identity hardening as performance work, not release-blocker.
- **F2/F3/F4**: SIMSOPT public docs do NOT specify the JAX normalization or Boost `dtmax` internals. These are source-parity findings, not public-doc violations.

---

## P0 — Correctness fixes (must ship before next release)

### Critical NaN / sentinel / validation bugs

- [ ] **F1 / H7 (EXPANDED post-validation)** — Zero-`m_maxima` NaN propagation, full scope:
    - (a) JAX `projection_l2_balls` at `src/simsopt/jax_core/pm_optimization.py:2122-2125`: replace `unit = m_maxima / m_maxima` with `jnp.ones_like(m_maxima)` AND replace `jnp.maximum` with `jnp.fmax`. **[Source: row-7 deeper report, audit-flagged HIGH H7]**
    - (b) CPU `projection_L2_balls` at `src/simsopt/solve/permanent_magnet_optimization.py:82`: same hazard (`np.sqrt(...) / mmax` then `np.maximum(1, ...)` propagates NaN). Use `np.fmax` and guard `mmax > 0`. **[Source: external validation critique + my live Read of the source code (not in committed audit reports); see provenance note below.]**
    - (c) CPU `prox_l1` at `src/simsopt/solve/permanent_magnet_optimization.py:63`: `np.abs(m) / mmax_vec` divides by zero. Add guard. **[Source: external validation + live source inspection; not in committed reports.]**
    - (d) JAX `prox_l1_jax` at `src/simsopt/solve/permanent_magnet_optimization_jax.py:320`: identical pattern `jnp.abs(matrix) / m_maxima[:, None]`. Add guard. **[Source: external validation + live source inspection.]**
    - (e) **Reachability**: `cell_vol = 0` for on-axis cylindrical cells at `src/simsopt/geo/permanent_magnet_grid.py:382` produces `m_maxima = B_max·cell_vol/μ₀ = 0`. **[Source: external validation + live source inspection.]**
    - Add regression tests covering all 4 entrypoints with `m_maxima = [1.0, 0.0, 1.0]` and zero-`m` rows. (3h total)

    **Provenance note for (b)-(e):** items (a) is from the audit reports (row-7 deeper, finding D1/H7). Items (b)-(e) were added during the post-checklist validation pass via direct source-code inspection, not from a per-row audit report. All four are reproducible bugs (live-verified by reading the named files and tracing the data flow), but they did NOT pass through the parallel-subagent audit process. Treat with appropriate confidence: bugs are real, but coverage of the broader bug class is the validation reviewer's call, not the original audit's.

- [ ] **F-DH2** — Replace `1e50` GPMO sentinel with `jnp.inf` at `pm_optimization.py:{609-611, 776-778, 1062-1065, 1597-1600}`. (LIVE-verified: collides with real costs at `‖b‖~1e26`; affects all 5 GPMO variants.) (30min)

- [ ] **F-DH3** — Add `nu > 0` validator at PM optimization API ingress. Current code at `pm_optimization.py:{2271, 2464}` produces silent NaN through `1/(2·0)`. (LIVE-verified.) (15min)

- [ ] **F-DH9** — Add shape assertion at `src/simsopt/objectives/integral_bdotn_jax.py:50` ingress. (LIVE-verified: `target.shape=(5,7)` silently broadcasts against `B.shape=(1,1,3)` and returns wrong `J=17.5`; CPU raises `RuntimeError`.) (30min)

- [ ] **F-DH1** — Add explicit `rcond` to RCLS `jnp.linalg.lstsq` at `src/simsopt/solve/wireframe_optimization_jax.py:149`: `rcond = max(LHS.shape) * jnp.finfo(LHS.dtype).eps`. (LIVE-verified: `diff_max ≈ 1.56e14` vs scipy on rank-deficient LHS.) Add rank-deficient regression test. (1h)

### (F2 moved to P7 — documentation-only correction, not a correctness blocker per post-validation review.)

---

## P1 — Tracing parity hardening (Row 4)

- [ ] **F3 / H2** — Add `dtmax` step ceiling. Edit 5 inline clamp sites in `src/simsopt/jax_core/tracing.py:{952, 1487, 1781, 2422, 2947}`: replace `jnp.minimum(h, tmax - t)` with `jnp.minimum(jnp.minimum(h, tmax - t), dtmax)`. Compute `dtmax = r0 * 0.5 * π / v_total` for particles, `r0 * 0.5 * π / AbsB` for fieldlines. Threaded as constructor arg. (1d)

- [ ] **F4 / H3** — Per-driver initial step heuristic. Replace `_INITIAL_STEP_FRACTION = 1/100` at `tracing.py:{218, 688-691}` with mode-specific recipe: `1e-3 * dtmax` for particle drivers, `1e-5 * dtmax` for fieldlines. Depends on F3. (4h)

- [ ] **F5 / H4** — Pass non-zero `atol` to `bracket_root_jax` matching the localizer's bracket-width target at `tracing.py:{1015, 1547, 2479, 3006}`. (Performance only; saves ~30 dead RHS evals per event.) (2h)

- [ ] **F-DH4** — Capture `phi_init` after first JAX integration step inside the criterion predicate at `tracing.py:{886, 1422, 2882}`. Current code captures from `y0` at construction time, diverging from C++ `iter==1` convention. (2h)

- [ ] **F-DH5** — Add `assert(s > 0)` equivalent in JAX Boozer GC RHS. Surface to driver as `status=-2`. Currently Boozer particles can silently spiral past axis with NaN derivatives. (2h)

- [ ] **F-DH6** — Add JAX-isolated mu/E conservation tests. Create `tests/jax_core/test_tracing_jax_conservation.py` with `|μ_t − μ_0| < 1e-10` and `|E_t − E_0| < 1e-10` over `t > T_bounce` for trapped orbits. Covers all Boozer GC modes (vacuum, no_k, full). (4h)

- [ ] **F-DH7** — Fix `bracket_root_jax` NaN poisoning at `tracing.py:765`. Replace unconditional `b - fb * width / (fb - fa)` with `jnp.where(jnp.abs(fb-fa) > 1e-300, false_position, bisection_midpoint)`. (1h)

- [ ] **F-H4-bracket** — Bracket-monotonicity enforcement at `tracing.py:711-712`. Docstring says `t_left <= t_right` is required but no internal check enforces it. All 4 internal callsites pass `(0.0, 1.0)` so unreachable today, but a future caller could trigger it. Add `t_left, t_right = jnp.minimum(...), jnp.maximum(...)` swap inside `bracket_root_jax`. (1h)

- [ ] **F-DH8** — Add explicit `t = tmax` back-fill at end of trajectory for non-terminated particles. Currently `loss_ctr` over-counts JAX losses vs C++. Match `tracing.cpp:441-444` `dense.calc_state(tmax, y)`. (2h)

- [ ] **F21 (Row 4 part)** — Fix stale docstring at `tracing.py:8-13` (falsely says fieldline RHS is `B/|B|`; actual code returns `B`). Remove stale "Boozer GC path deferred" comment at `tracing.py:386-388`. (15min)

---

## P2 — Interpolation boundary semantics (Row 5)

- [ ] **F6 / H5** — Cell-locator: replace `jnp.floor(...).astype(jnp.int32)` with `jnp.trunc(...).astype(jnp.int32)` at `src/simsopt/jax_core/regular_grid_interp.py:510-512`. (Single-character change after corrigendum; the in-bounds gate already works.) Add regression test at `nx=2, x = xmin - 0.1*hx`. (30min)

- [ ] **F7 / H6** — API decision required. Three options, pick one: (a) add optional `existing_result` argument to `evaluate_local` and use `jnp.where(in_kept_cell, result, existing)`; (b) accept JAX-only stricter contract, document, **remove the parity claim** from the ledger; (c) wrap at consumer side. Update `tests/jax_core/test_regular_grid_interp_item13.py:186-196` accordingly. (decision + 4h)

- [ ] **F-DH14 (sharpened from critique)** — `_DeviceSpec` cache invalidation at `src/simsopt/jax_core/interpolated_field.py:346-381`. The `frozen=True` dataclass references mutable NumPy arrays; id-keying alone does NOT help because same-id mutation defeats it. Correct fix: deep-copy AND set `array.flags.writeable = False` on construction so in-place mutation raises. Currently mutating underlying `spec.cell_table` silently produces stale results across consumers. (3h)

- [ ] **F-DH15** — Promote int32 → int64 for flat cell-index arithmetic at `regular_grid_interp.py:{510-512, 525}` and `src/simsoptpp/regular_grid_interpolant_3d.h:124-127`. UB in C++ at `nx=ny=nz=1000, degree=4` (`6.4e10 > INT_MAX`). (2h)

- [ ] **F-DH16** — Add `degree=4` cross-oracle test on polynomially-exact fixture (`nx·degree` knots interpolating degree-`(nx·degree-1)` polynomial → residual = 0 in exact arithmetic). Bare-kernel `test_cpp_cross_oracle` currently covers only `degree in [1,2,3]`. (2h)

- [ ] **F-D5 (Row 5 deeper)** — Add silent-wrong-physics warning when `jax.grad` flows through `evaluate_batch`. JAX returns gradient of piecewise polynomial, NOT the `GradAbsB` interpolant value. Add either a docstring warning or a custom_vjp that raises. (decision + 1h)

---

## P3 — Upstream coordinated fixes (C++ + JAX in one PR)

- [ ] **F9 / H9** — Wireframe GSCO operator precedence. Patch BOTH together: `src/simsoptpp/wireframe_optimization.cpp:270` change `(opt_ind + nLoops % (twoNLoops))` to `((opt_ind + nLoops) % twoNLoops)`; mirror at `src/simsopt/solve/wireframe_optimization_jax.py:406`. Add regression test exercising negative→positive undo direction. Survey downstream consumers (CIEMAS) before landing. (1d)

- [ ] **F10 / CT-4** — Boozer radial OpenMP race. Apply correct fix at `src/simsoptpp/boozerradialinterpolant.cpp:{147-156, 165-175}` using scalar local accumulators:
    ```cpp
    for (int im = 1; im < num_modes; ++im) {
        double sum = 0.0, norm = 0.0;
        #pragma omp parallel for reduction(+:sum, norm)
        for (int ip = 0; ip < num_points; ++ip) {
            double s = sin(xm(im)*thetas(ip) - xn(im)*zetas(ip));
            sum  += K(ip) * s;
            norm += s * s;
        }
        kmns(im) = sum / norm;
    }
    ```
    (NOT `reduction(+:kmns, norm)` — OpenMP doesn't accept xtensor array types.) Same pattern for `fourier_transform_even`. (2h)

- [ ] **F-DH10b (Row 12 deeper)** — Add closed-form NumPy oracle test for forward `fourier_transform_*` that does NOT call simsoptpp (insulates JAX gate from future OMP regressions). (2h)

- [ ] **F-DH23** — Replace non-ISO VLA at `src/simsoptpp/magneticfield_wireframe.cpp:39-40` with `std::vector<double*> halfPrd_ptr(nHalfPrds)` and `std::vector<double> seg_signs(nHalfPrds)`. Stack-overflow vector at pathological `nHalfPrds`. (1h)

- [ ] **F-DH24** — Initialize `int opt_ind_prev = -1;` at `src/simsoptpp/wireframe_optimization.cpp:154`. Currently UB-protected only by synchronized `break` after `stop_none_eligible`. JAX already does this correctly. (15min)

- [ ] **F-DH (Row 13)** — Move loop-body local variables INSIDE `#pragma omp parallel for` region at `src/simsoptpp/dommaschk.cpp:{475-476, 503-504}`. Textbook data-race pattern; explains any flakiness under `OMP_NUM_THREADS > 1`. (30min)

- [ ] **F-DH17** — Dipole-on-axis with non-cartesian `coordinate_flag` silent C++↔JAX divergence. **(LIVE-verified.)** With a dipole at `(0,0,0)` and `coordinate_flag="cylindrical"` or `"toroidal"`, C++ returns `[NaN, NaN, finite-z]` while JAX returns finite values. Root cause: `xsimd::atan2(0, 0)` returns NaN at `src/simsoptpp/dipole_field.cpp:332-334` and propagates through rotation factors; `jnp.atan2(0, 0)` returns 0 (matches `std::atan2`) at `src/simsopt/jax_core/dipole_field.py:458`. JAX also silently treats unknown `coordinate_flag` typos (e.g., `"sphereical"`) as cartesian via the fall-through `else` at `:466-468`. Fix: guard `atan2(0, 0)` with sentinel in C++ to match JAX (recommended), or document that dipoles must not lie on axis for non-cartesian frames. Add typo regression test. (2h)

- [ ] **F-DH25 (sharpened from critique)** — Wireframe unsafe int32 narrowing. The risk is not just `n_segments > INT32_MAX` (count) but also **segment index VALUES** exceeding int32 (a wireframe with many nodes where edge indices are >2³¹ even when count fits). Both fail. `wframe.segments` is int64 (`src/simsopt/geo/wireframe_toroidal.py:163`); narrowed at `wireframefield_jax.py:26` AND `wireframe.py:115` AND in any CPU wrapper that passes int64 arrays to pybind through similar boundaries. Audit all int32 casts of `segments`/`seg_signs`/`nodes`-derived arrays; either promote to int64 throughout, or check both `n_segments` AND `max(segments) > INT32_MAX` at the boundary. (2h)

---

## P4 — Autodiff NaN cliffs (NEW priority tier, CT-D1)

Apply the **double-where idiom** at each site:
```python
safe = jnp.where(cond, x, ones_like(x))
result = jnp.where(cond, f(safe), fallback)
```

- [ ] **F-DH10** — `"local"` flux gradient guard at `src/simsopt/objectives/integral_bdotn_jax.py:65-78`. Currently primal returns `+inf` at `|B|=0` but gradient is finite-on-survivors (silent failure mode through bare kernel). (1h)

- [ ] **F-DH19** — `_unitnormal` autodiff fix at `src/simsopt/geo/surface_fourier_jax.py:1114-1115` and `src/simsopt/jax_core/surface_rzfourier.py:680-683`. (LIVE-verified at `n=(1e-300, 0, 0)` returns `[inf, NaN, NaN]` forward and all-NaN Jacobian.) (2h)

- [ ] **F-DH20** — Quaternion normalization at `src/simsopt/geo/curveplanarfourier.py:17-22`. Forward value at `q=0` is correct (identity rotation), but `jax.grad` w.r.t. quaternion DOFs returns NaN where C++ analytic returns 0. Add gradient parity test at `q=0`. (1h)

- [ ] **F-DH (Row 3)** — `_inverse_modB` gradient at `|B|=0` in `src/simsopt/geo/boozer_residual_jax.py`. Currently returns `+inf` in primal AND gradient; document or guard. (1h)

- [ ] **Lint rule** — Add project lint rule searching for `jnp.where(.*0.*, .*x/0.*, ...)` patterns; flag for double-where idiom review. (2h)

---

## P5 — Stale-state contracts (NEW priority tier, CT-D4)

For every JAX adapter exposing CPU-mutable attributes, choose ONE:
- Deep-copy at construction, ban mutation;
- Re-snapshot on every entry;
- Provide explicit `invalidate_cache()` documented in public API.

- [ ] **F-DH (Row 5)** — `_DeviceSpec` cache (covered by F-DH14 above; cross-listed). (—)

- [ ] **F-DH (Row 8)** — `dB_by_dcoilcurrents` invalidation on `set_points_*` at `src/simsopt/field/wireframefield_jax.py:{60-70, 116-119}`. Currently latent (always overwritten); harden before regression. (2h)

- [ ] **F-DH (Row 12)** — `bri.psi0` mutability vs K-spline factor at construction (line 704). Post-construction `psi0` mutation desyncs JAX wrapper. Document as immutable or invalidate. (1h)

- [ ] **F-DH (Row 13, D-5)** — `DommaschkJAX.coeffs` capture vs CPU re-read at `src/simsopt/field/dommaschk_jax.py:79`. CPU `Dommaschk` re-reads `self.coeffs` per call; JAX uses construction-time snapshot — silent divergence under mutation. (2h)

- [ ] **F-DH (Row 7)** — `pm_optimization` shape-recompile cost. Every change in `N` or `P` triggers ~100ms JAX recompile; not shape-polymorphic. Add shape-polymorphism or document. (4h)

---

## P6 — Test oracle hardening

### Coverage gaps from first-pass

- [ ] **F11** — Add zero-`m_maxima` regression test (gates F1).
- [ ] **F11b** — Add L0/L1 kernel-level parity assertions (currently only diagnostic-reporting parity for PM).
- [ ] **F11c** — Add `single_direction` coverage for `GPMO_backtracking` JAX (Row 7 INFO).
- [ ] **F12a** — Boozer T2: add direct unit-level oracle test for `boozer_residual_scalar_and_grad_cpu_ordered` against `sopp.boozer_residual_ds`.
- [ ] **F12b** — Boozer T3: add `boozer_residual_jacobian_composed` C++ Jacobian oracle.
- [ ] **F12c** — Boozer T4: add `jax.hessian(boozer_penalty_composed)` C++ Hessian oracle.
- [ ] **F12d** — Boozer T5: add `boozer_residual_coil_vjp(weight_inv_modB=True)` path coverage.
- [ ] **F13a** — Dipole autodiff w.r.t. moments (linear). FD-cross-validated.
- [ ] **F13b** — Dipole autodiff w.r.t. positions (nonlinear).
- [ ] **F13c** — Dipole autodiff w.r.t. evaluation points.
- [ ] **F13d** — Dipole singularity-policy docstring warning in `dipole_field.py` and C++ header.
- [ ] **F14a** — Wireframe RCLS test with rank-deficient `LHS` (covered by F-DH1; cross-listed).
- [ ] **F14b** — Wireframe large-fixture GSCO parity at `nGrid=200, nLoops=50` (current `nGrid=5, nLoops=2` below realistic scale).
- [ ] **F14c** — Wireframe `bnorm_obj_matrices_jax` with `ext_field` / `bnorm_target` branches.
- [ ] **F15a** — Surface XYZ-Fourier `dgammadash1_by_dcoeff_impl` column-by-column C++ oracle.
- [ ] **F15b** — Surface XYZ-Fourier `dgammadash2_by_dcoeff_impl` column-by-column C++ oracle.
- [ ] **F15c** — Surface `dnormal_by_dcoeff_vjp` parity test.
- [ ] **F16a** — Extend `tests/geo/test_curve_item05_closeout.py::test_curve_spec_pullback_production_scale_parity` to pin `gammadash` and `gammadashdash` for RZ Fourier and Planar Fourier.
- [ ] **F16b** — Direct `pair_linking_number_pure` test against `simsoptpp.compute_linking_number` on dense fixture.
- [ ] **F17a** — Closed-form NumPy oracle for `fourier_transform_odd/even` (covered by F-DH10b; cross-listed).
- [ ] **F17b** — Direct `_compute_K_per_point` parity test (currently transitive).
- [ ] **F18a** — Direct `div(B) = trace(dB) = 0` assertion on JAX Dommaschk output.
- [ ] **F18b** — Central-FD Taylor test for `dommaschk_dB` (Reiman has one already).
- [ ] **F18c** — Odd-`k` Reiman regression (existing test uses `k=6` even, masking arctan vs arctan2 divergence).
- [ ] **F18d** — `jax.grad`-over-coefficients parity test (Dommaschk + Reiman).
- [ ] **F18e** — Explicit `B_φ = -1` assertion for Reiman.

### Coverage gaps from deeper-pass

- [ ] **F-DH11** — Surface-DOF drift guard regression test. Construct `SquaredFluxJAX`, mutate underlying surface DOFs, assert `_raise_if_surface_dofs_drifted` at `fluxobjective_jax.py:378-386` fires. (1h)

- [ ] **F-DH12** — `_split_decision_vector` bounds check. Add assertion `surface_size >= 0` at `boozer_residual_jax.py:73-82`. (15min)

- [ ] **F-DH13** — Delete duplicated G-from-currents formula at `boozer_residual_jax.py:554-556`; import SSOT `compute_G_from_currents` from `label_constraints_jax.py:49-61`. Three different inline spellings exist; algebraically identical today but no guard against drift. (30min)

- [ ] **F-DH21 (HIGH severity, re-tagged per Pass-4)** — `B·∇ζ = G(s)` Boozer identity test. HIGH-severity coverage gap: tested in NEITHER item 32 nor item 33; both backends could agree on the same bug. Add to `tests/field/test_boozermagneticfield_jax_item33.py`. (2h)

- [ ] **F-DH22** — `mn_factor` extrapolation test at `s < s_half_mn[0]`. Clarify docstring (`s^{-m/2}` not `s^{m/2}`). (1h)

- [ ] **F-DH (Row 12)** — `enforce_qs=True` and `enforce_vacuum=True` paths never tested; add coverage. (2h)

- [ ] **F-DH (Row 5)** — `value_size != 1, 3` cross-oracle gap; test at `value_size in [5, 7, 8]` (SIMD-padded C++ path). (2h)

- [ ] **F-DH (Row 10)** — High-resolution surface stress test at `mpol=ntor=10` for stellsym scatter, normal, area, volume. Current tests max at `mpol≤2, ntor≤2`. (2h)

- [ ] **F-DH (Row 10)** — Non-stellsym `d2volume/d2area` Hessian validation at `mpol > 2` against C++ SIMD-vectorised oracle. (2h)

- [ ] **F-DH (Row 11)** — Non-identity quaternion Planar curve production fixture (current `q ≈ (1, ε, ε, ε)` never exercises hard rotation). (1h)

- [ ] **F-DH (Row 4)** — Lost-particle classification edge cases: exit on quadrature face, two stopping criteria fire same step, levelset returns exactly 0. (4h)

---

## P7 — Documentation / API surface cleanup

- [ ] **F21a** — Tracing module docstring at `src/simsopt/jax_core/tracing.py:8-13` falsely says fieldline RHS is `B/|B|` (covered in P1; cross-listed). (—)

- [ ] **F21b** — Dipole singularity-policy docstring (covered by F13d; cross-listed). (—)

- [ ] **F22a** — `wireframe_segment_dB_by_dX_contributions` (Row 8 INFO): exported but no direct C++ parity test and no in-tree consumer. Either add test or remove export. (1h)

- [ ] **F22b** — `residual_BdotN`, `signed_BdotN_flux` (Row 2): JAX-only public exports with no per-point oracle. Add closed-form NumPy per-point oracle. (2h)

- [ ] **F-DH26** — Fix `_simsopt_jax_native_field = True` over-claim at `src/simsopt/field/wireframefield_jax.py:47`. No `B_vjp` implemented → `AttributeError` for `MagneticFieldSum([WireframeFieldJAX, BiotSavartJAX]).B_vjp(v)`. Either implement `B_vjp` or set flag to `False`. (decision + 2h)

- [ ] **F-DH (Row 6)** — Fix SIMSOPT-wide `dA_by_dX` pybind docstring at `src/simsoptpp/python_magneticfield.cpp:45`. Claims `∂_j A_l` but actual storage is `dA[p, j, k] = ∂A_j/∂x_k` (transposed). Load-bearing for downstream `dA_by_dX` consumers. (1h)

- [ ] **F2 (moved from P0, post-validation)** — Correct module docstring at `src/simsopt/geo/boozer_residual_jax.py:31-34`. Replace "matching the C++ normalization" with: "JAX normalizes by `1/(3·nphi·ntheta)`; raw C++ symbol `sopp.boozer_residual` does NOT carry this normalization; CPU production path normalizes inline at `boozersurface.py:601-602`." Documentation-only — production scalar paths are equivalent per first-pass corrigendum. (30min)

- [ ] **F-DH (Row 6 validation contract)** — Asymmetric input-validation contracts. JAX `dipole_field.py:439` raises `ValueError` on `unitnormal.shape != points.shape`; C++ `dipole_field.cpp:310` silently reads `unitnormal(i+k, d)` with no shape check (OOB-read potential). Both backends also silently accept unknown `coordinate_flag` strings. Either align (recommended: C++ adds a shape assertion that raises) or document the asymmetry. (1h)

- [ ] **F-DH (Row 3)** — Reconcile default `weight_inv_modB` across sibling APIs. `boozer_residual_coil_vjp` and `_boozer_residual_vector_composed` default `False`; `boozer_residual_scalar`/`vector`/`penalty_composed` default `True`. (30min)

- [ ] **F-DH (Row 5)** — Document `_DeviceSpec` lifetime and memory pressure for production grids. (30min)

- [ ] **F-DH (Row 8)** — Document `dB[p, k, m]` (component-first) layout convention divergence from CLAUDE.md abstract `[p, j, l]`. Matches C++ storage exactly; no numerical bug, but readers will trip. (15min)

- [ ] **F-DH (Row 11)** — `RotatedCurve` not exposed to JAX `curve_geometry` dispatcher (`curve_spec_from_curve` raises `NotImplementedError`). Either add support or document the limitation. (decision + 1d if implementing)

---

## P5+ — Performance follow-ups

- [ ] **F19 / H8 (corrigendum)** — `boozer_residual_jacobian_composed` duplicate forward pass at `src/simsopt/geo/boozer_residual_jax.py:738-739`. Switch to `jax.linearize` or `value_and_jacfwd`. ~2× speedup. (2h)

- [ ] **F20a (CT-1, Row 2 M-1) — REPHRASED per critique** — `"normalized"` flux byte-identity is NOT byte-identical to C++ symmetric reduction, but the algebra is equivalent and the AD-uniform per-point form is load-bearing. Source report explicitly recommends: document in CLAUDE.md AND optionally add a strict-oracle path keyed off a `reduction_mode="strict_oracle"` kwarg, NOT replace the algebra by default. Land the docstring + CLAUDE.md note; the strict-oracle path is optional and only if a future byte-id gate demands it. (1h)

- [ ] **F20b (CT-1, Row 5)** — `jnp.einsum` reduction tightening. XLA-chosen order vs C++ k-fastest hand-rolled FMA. Switch to `lax.scan`/`lax.fori_loop` if byte-identity becomes binding. (1d)

- [ ] **F20c (CT-1, Row 8)** — `wireframe_segment_B_contributions` and `_dB_by_dX_contributions` use `jnp.sum(jax.vmap(...))` (tree reduction); switch to `lax.scan` to match C++ sequential `axpy_array`. (4h)

- [ ] **F20d (CT-1, Row 13)** — `_accumulate_terms` intentional monomial merge breaks ULP at coefficients ≥ 1e10. Document the limit. (30min)

- [ ] **F-DH (Row 10)** — `_scatter_matrix` dense allocation at `surface_rzfourier.py:327-337`. Replace with `lax.scatter` to match XYZ-tensor path at `surface_fourier_jax.py:1203-1234`. ~88MB allocation per call at `mpol=ntor=20`. (4h)

- [ ] **F-DH (Row 10)** — `jax.hessian(surface_volume_from_dofs)` at `mpol=ntor=20, nphi=ntheta=32` produces 64GB tensor. Add memory guard. (1h)

- [ ] **F-DH (Row 7)** — Make PM optimization shape-polymorphic to avoid recompile on `N`/`P` change (cross-listed in P5). (4h)

---

## Pass-4 additions — missing items recovered from row reports

The first checklist consolidated heavily. The 4th validation pass identified ~67 items present in the per-row reports but not represented as discrete checkboxes. Adding them here grouped by row, each tagged with its natural priority tier.

### Row 02 (integral B·n) — missing

- [ ] **R02-A1** [P6] — Add `coil_current_fixed_geometry_value_and_grad_jax` vs `SquaredFlux.dJ()` direct gradient parity test (currently FD-only). (2h)
- [ ] **R02-A2** [P0/P1] — Empty-mesh parity divergence: CPU returns `nan` for `nphi=0`, JAX returns `0.0`. Decide canonical behavior, document, add regression. (1h)
- [ ] **R02-A3** [P4] — `target=None` zero-target NaN-poison through `normal` arithmetic at `objectives_flux.py:38-42` (`jnp.sum(normal) - jnp.sum(normal)` produces NaN if any normal is NaN). Add explicit zero-target path. (1h)
- [ ] **R02-A4** [P7] — Float32 / complex public-kernel contract tests + docstring. Currently silent dtype passthrough. (1h)
- [ ] **R02-A5** [P6] — Production-scale (64×64) byte-identity test (closeout caps at 16×8). (1h)

### Row 03 (Boozer residual) — missing

- [ ] **R03-A1** [P3] — `optimize_G=False` fixed-G semantic gap: JAX has no path to pin user-supplied G and exclude from decision vector. Add or document refusal. (2h)
- [ ] **R03-A2** [P5+] — Add `n_res < n_dofs ⇒ jacrev` heuristic to `boozer_residual_jacobian_composed` to complement F19's value+jacfwd refactor. (2h)
- [ ] **R03-A3** [P0/P7] — Float64 input-contract assertion at `boozer_residual_scalar` and `_as_runtime_float64` (currently silently promotes/demotes). (1h)
- [ ] **R03-A4** [P7] — C++ SIMD-load invariant comment at `boozerresidual_impl.h:205-208` (AlignedPaddedVec padding assumption); add CI probe for non-SIMD branch. (1h)
- [ ] **R03-A5** [P7] — Boozer `dB_by_dX` 4D layout docstring (matches CLAUDE.md `[p, j, l]` abstract convention). (30min)

### Row 04 (tracing) — missing

- [ ] **R04-A1** [P6] — Fixed-seed `loss_ctr` parity test C++ vs JAX (gates F-DH8 at its broad scope; exposes the structural bias for cleanly-stopped trajectories). (2h)
- [ ] **R04-A2** [P1] — `accepted_count == max_steps` hard-error/structured warning at orchestrator level (currently silent `logger.debug` only). (1h)
- [ ] **R04-A3** [P6] — Levelset phi-wraparound CPU↔JAX boundary-parity tests (`[0, 2π)` reduction asymmetry between RHS-path `_continuous_phi` unwrap and classifier Cartesian sampling). (2h)
- [ ] **R04-A4** [P6] — `_continuous_phi` / `get_phi` exact-edge tests at `phi = ±π` and `phi_init = π` with rounding probes. (1h)
- [x] **F-DH8 scope reconciled** [meta] — F-DH8 was originally listed with broad scope (P1, line 79); Pass-3 corrigendum narrowed it; Pass-4 re-broadened. **Current state: F-DH8 at line 79 stands at its original broad scope.** Corrigendum at line 24 reflects this. No separate action item — this entry pre-checked as a tracking note.

### Row 05 (regular-grid interp) — missing

- [ ] **R05-A1** [P6] — Extend `degree=4` cross-oracle (F-DH16) to ALSO cover `degree=5` and add high-magnitude-cell stress fixture. (1h)
- [ ] **R05-A2** [P6] — NaN-input parity contract test (turning-point banana orbit pumps NaN coordinates into interpolant; verify finite-or-NaN output is consistent). (1h)
- [ ] **R05-A3** [P6] — Stellsym/nfp fold boundary test (per fold subtlety where roundoff drives phi negative). (2h)
- [ ] **R05-A4** [P6] — Chebyshev-node cross-oracle test (currently only uniform). (1h)
- [ ] **R05-A5** [P6] — `estimate_error` cross-oracle pinning (acknowledge RNG divergence; assert polynomial-exactness case agrees). (1h)
- [ ] **R05-A6** [P6] — `value_size in [5, 7, 8]` SIMD-padded-tail invariant test. (1h)
- [ ] **R05-A7** [P7] — Sparse-skip-cell-map sparse-vs-dense benchmark for JAX sentinel-row redirect vs C++ unordered_map. (4h)

### Row 06 (dipole) — missing

- [ ] **R06-A1** [P5+] — C++ Bn hoist redundant `mp_phi_new`/`sphi0`/`cphi0` computations from inner loop at `dipole_field.cpp:320-338`. ~10× speedup for stellsym=1,nfp=5. (2h)
- [ ] **R06-A2** [P5+] — Replace `pow(-1, stell)` with branchless `(1 - 2*stell)` in C++ Bn loop. (15min)
- [ ] **R06-A3** [P3] — Replace `1e5` grid-distance sentinel at `dipole_field.cpp:776-777` with `std::numeric_limits<double>::infinity()`. Breaks for device-distances > 316m. (15min)
- [ ] **R06-A4** [P7] — Mixed-dtype upcast documentation (both backends silently upcast float32→float64). (15min)
- [ ] **R06-A5** [P5+] — Tighten F-DH17 action: choose+document JAX↔C++ on-axis alignment (recommended: C++ guards `atan2(0,0)` to match JAX). Add `coordinate_flag` whitelist validation (reject typos) at JAX dispatcher. (1h)

### Row 07 (PM optimization) — missing

- [ ] **R07-A1** [P0] — `alpha > 0` validator at PM optimization API ingress (sister to F-DH3 `nu > 0`). (15min)
- [ ] **R07-A2** [P6/P7] — Too-large-alpha oscillation regression: `alpha > 2/lambda_max(H)` causes wild oscillation, not divergence. Add diagnostic warning and test. (2h)
- [ ] **R07-A3** [P0] — `jnp.isfinite` guard before all 5 GPMO `argmin` callsites (`pm_optimization.py:{625, 792, 1267, 1617, 1915}`). `argmin` treats NaN as smallest; one NaN leak corrupts greedy selection. (1h)
- [ ] **R07-A4** [P6] — `mwpgp_step` expand-branch execution coverage test (currently only `cond` count is asserted). (2h)
- [ ] **R07-A5** [P6] — Non-uniform `m_maxima` penalty regression (penalty index-quirk faithfully mirrored from C++; should be pinned). (1h)
- [ ] **R07-A6** [P6] — End-to-end `relax_and_split_jax` vs CPU after multiple outer iterations. (4h)
- [ ] **R07-A7** [P7] — Document MwPGP early-stop/history-trap: JAX runs fixed-step; downstream callers expecting `objective_history`/`m_history` from C++ get nothing. (30min)

### Row 08 (wireframe field) — missing

- [ ] **R08-A1** [P0/P7] — Public-API `points.ndim == 2 and shape[-1] == 3` validation at `wireframe.py:133, 188-195`. Currently silent mis-broadcast via `factor[:, None]`. (1h)
- [ ] **R08-A2** [P7] — `dB_by_dsegmentcurrents(compute_derivatives)` semantic ambiguity: both CPU and JAX wrappers return B contributions regardless of argument; docstring promises something else. Existing parity test passes vacuously. (1h)
- [ ] **R08-A3** [P6] — Singular-regime contract test: query on/near a wire segment. Currently no defensive floor; document expected NaN/inf behavior. (1h)
- [ ] **R08-A4** [P5] — Snapshot `_n_segments` at construction (currently live access to `self.wireframe.n_segments` creates stale-snapshot divergence point). (1h)
- [ ] **R08-A5** [P6] — Realistic-size slow-fixture wireframe test (`n_segments > 2^15` topology). (4h)
- [ ] **R08-A6** [P6] — Autodiff-through-currents linearity invariant test. (1h)

### Row 09 (wireframe optimization) — missing

- [ ] **R09-A1** [P3] — Port QR NaN-retry safeguard from CPU `wireframe_optimization.py:824` to JAX `wireframe_optimization_jax.py:130`. (1h)
- [ ] **R09-A2** [P6] — RCLS parity test via `ToroidalWireframe.set_poloidal_current` / `set_toroidal_current` / `set_segments_constrained`. (2h)
- [ ] **R09-A3** [P7] — Document `default_current ≈ tol` edge-case behavior. (30min)
- [ ] **R09-A4** [P6] — Explicit `stop_none_eligible` / undo-branch coverage gap (currently never exercised in tests). (1h)
- [ ] **R09-A5** [P3] — Refactor C++ sticky stopping-flag hygiene at `wireframe_optimization.cpp` (`accept_current_loop`/`stop_none_eligible`/`stop_undone_loop` declared outside loop, never reset; latent bug masked by `break`). (1h)

### Row 10 (surfaces) — missing

- [ ] **R10-A1** [P7] — Document RZ-derivative-path `~1e-12` cross-machine reduction-order tolerance lane explicitly. (30min)
- [ ] **R10-A2** [P6] — Non-tensor `SurfaceXYZFourier::dnormal_by_dcoeff` / `d2normal_by_dcoeffdcoeff` direct C++ oracle (currently no column-by-column parity). (4h)
- [ ] **R10-A3** [P5+] — Tighten F-DH18 memory claim: 88MB is at extreme (`mpol=ntor=20`); source supports smaller values typical (~few MB). Rephrase as "scales unfavorably," not "88MB per call." (15min)
- [ ] **R10-A4** [P6] — Near-zero-normal autodiff tests for `darea_by_dcoeff` / `dvolume_by_dcoeff` (sister to F-DH19). (1h)
- [ ] **R10-A5** [P6] — High-resolution transfer-guard memory regression test (mpol=ntor≥10). (2h)
- [ ] **R10-A6** [P5+] — `_block_mode_positions` dedupe (currently O(modes²) hot path). (4h)
- [ ] **R10-A7** [P6] — Production-scale non-stellsym / extreme-nfp round-trip test. (2h)

### Row 11 (curves) — missing

- [ ] **R11-A1** [P3] — C++ off-by-one at `src/simsoptpp/curveplanarfourier.cpp:411`: `i < 2` should be `i < 3` (matches siblings at 281/543/696). Currently benign (zero-init), but typo for robustness. (15min)
- [ ] **R11-A2** [P6] — Expand F16a: also pin `gammadashdashdash` and use a hard nontrivial quaternion fixture (current Planar fixture `q ≈ (1, ε, ε, ε)` is near-identity). (1h)
- [ ] **R11-A3** [P7] — `pair_linking_number_pure` non-differentiability docstring (returns `int32` after `jnp.round`; not safe for `jax.grad`). (15min)
- [ ] **R11-A4** [P7] — Curvature/torsion NaN-contract documentation at inflection points (`|γ'|=0` or `γ'×γ''=0`). (30min)

### Row 12 (Boozer radial) — missing

- [x] **R12-A1 (resolved in this pass)** [meta] — DH21 severity re-tagged from CRITICAL to HIGH at F-DH21 line 212. No further action.
- [ ] **R12-A2** [P5] — F-DH (Row 12) stale-state scope expansion: add `enforce_qs`, `enforce_vacuum`, `N`, `no_K`, and mode-array mutation to the post-construction-mutability audit. (2h)
- [ ] **R12-A3** [P6] — `num_modes == 1` edge case, empty arrays, mixed-precision regression, GPU deterministic-lane, JIT-without-host-roundtrip tests. (4h)
- [ ] **R12-A4** [P7] — Positional-indexing maintenance hardening (current `kmns(im)` / `kmnc(im)` positional access is fragile to mode-table reorder). (1h)
- [ ] **F-DH10b mistiered** [P6] — Move from P3 (upstream) to P6 (test oracles): it's a JAX-side closed-form NumPy oracle test, not a coordinated C++/JAX fix. (—)

### Row 13 (analytic fields) — missing

- [ ] **R13-A1** [P6] — F20d under-specified: source asks for explicit guarded fixture/skip OR relaxed-kernel lane, not a 30min doc note. Bump effort and scope. (2h)
- [ ] **R13-A2** [P4/P7] — `jax.grad(reiman_B)` NaN-at-axis-ring / arctan2 branch-cut documentation and tolerant test (gradient diverges as expected; pin tolerance). (1h)
- [ ] **R13-A3** [P0/P3] — Reiman `k_theta >= 1` validation. `k=0` produces singular `rpow_m4 = rmin^{-4}`. (15min)
- [ ] **R13-A4** [P7] — Dommaschk default-constructor / `n=0, Z=0` discrepancy: C++ `Nmn(m, -1)` and JAX `_nmn_terms(m, -1)` produce 0 via different paths; document. (30min)
- [ ] **R13-A5** [P3] — Analytic-fields public-API point-shape validation (both backends silently accept wrong-shape `b` arrays). (1h)
- [ ] **R13-A6** [P5+] — Bound/expose-clear action for unbounded `lru_cache` in JAX Dommaschk polynomial expansion. (1h)

---

## P8 — Out-of-scope but flagged

The audit did NOT cover these areas; future work:

- [ ] **Integration-level testing**: Stage-1+Stage-2 composition; single-stage with all modules engaged.
- [ ] **CUDA/GPU determinism**: reduction order, atomicAdd, FP determinism on device.
- [ ] **Production-scale benchmarks**: `nphi≈64, ntheta≈64, mpol≈10, ntor≈10, ncoils≈50, npoints≈10^5`.
- [ ] **Cross-platform parity**: macOS Apple Silicon vs Linux x86_64 vs GPU; LS path documented non-portable.
- [ ] **Long-trajectory robustness**: tracing over `tmax ≈ 1 sec` (~10^9 RHS evaluations).
- [ ] **Concurrent-call safety**: behavior under `multiprocessing` / `dask` parallelism.

---

## Summary (post-validation, corrected)

| Tier | Count | Effort estimate |
|------|------:|----------------|
| P0 (correctness, must-ship — F1 now expanded across 4 entrypoints + reachability) | 5 | ~5-6 hours |
| P1 (tracing) | 10 | ~3 days |
| P2 (interp boundary) | 6 | ~2 days |
| P3 (upstream coordinated, +DH17 +DH25) | 8 | ~2 days |
| P4 (autodiff NaN cliffs) | 5 | ~1 day |
| P5 (stale-state contracts) | 5 | ~2 days |
| P6 (test oracles) | 37 | ~7 days |
| P7 (docs/API, +F2 moved here +validation-contract) | 12 | ~7 hours |
| P5+ (performance) | 8 | ~3 days |
| P8 (out-of-scope) | 6 | — |
| Pass-4 additions (recovered from row reports) | 65 unchecked + 2 meta-resolved | ~1.5 weeks |
| **TOTAL ACTIONABLE (unchecked)** | **161** | **~5 weeks** |
| **TOTAL UNCHECKED (incl P8)** | **167** | — |
| Pre-checked meta entries (scope-reconciliation tracking) | 2 | — |
| **GRAND TOTAL (`- \[ \]` + `- \[x\]`)** | **169** | — |

(Per-tier counts verified via `sed -n '/^## TIER/,/^## NEXT/p' | grep -c "^- \[ \]"`. Sum: 5+10+6+8+5+5+37+12+8+65+6 = 167 ✓ matches `grep -c "^- \[ \]"`. Adding 2 `[x]` meta entries = 169 grand total.)

**MUST-SHIP path:** check off all 5 P0 items (~5-6 hours total — F1 now expanded to cover CPU `projection_L2_balls`, CPU+JAX `prox_l1`, plus PM-grid `cell_vol=0` reachability). The rest can be tiered into release planning.

**Note on the deeper-pass synthesis count:** the deeper synthesis text says "19 new HIGH" but its per-row census table totals 26. The 26 figure is authoritative; the "19" in the prose is stale from an earlier draft.

---

## Source artifacts

- First-pass synthesis: `.artifacts/parity_audit_2026-05-16/00_SYNTHESIS.md`
- First-pass per-row reports: `.artifacts/parity_audit_2026-05-16/{02..13}_*.md`
- Deeper-pass synthesis: `.artifacts/parity_audit_2026-05-16/deeper/00_SYNTHESIS_DEEPER.md`
- Deeper-pass per-row reports: `.artifacts/parity_audit_2026-05-16/deeper/{02..13}_*_DEEPER.md`
