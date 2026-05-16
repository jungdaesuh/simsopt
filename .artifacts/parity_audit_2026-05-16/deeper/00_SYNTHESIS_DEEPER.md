# SYNTHESIS — JAX↔C++ Parity Deeper Audit (Second Pass)

**Audit timestamp:** 2026-05-16
**Method:** 12 parallel max-effort Opus 4.7 subagents, second pass. Each subagent received the first-pass headline and was directed to hunt for issues a forward-formula audit would systematically miss: autodiff NaN cliffs, dtype/shape silent broadcasts, JIT-closure stale state, C++ undefined behavior, API surface drift, test-coverage gaps at production scale, sentinel collisions, identity-level invariants.

**Total written:** 6605 lines of detailed second-pass audit (12 reports, ~384 KB) plus this synthesis.

**Top-line:** The first pass verified forward formula parity. The second pass found that forward parity does NOT imply production safety. **19 new HIGH-severity findings** surfaced — almost all are autodiff cliffs, silent validation gaps, stale-state risks, sentinel collisions, or untested invariants. None contradict the first pass's formula verdicts; they expose modes of failure orthogonal to "does the formula match."

---

## Severity census (second-pass new findings, not counting first-pass items)

| Row | Module | HIGH (new) | MED (new) | LOW (new) | Headline novel finding |
|----:|--------|-----------:|----------:|----------:|------------------------|
| 2 | `integral_bdotn_jax` | **3** | 4 | 6 | Silent target-shape broadcast (no validation) → wrong J; `"local"` returns `inf` with finite gradient (silent) |
| 3 | `boozer_residual_jax` | **2** | 4 | 3 | `_split_decision_vector` no bounds check; G-from-currents formula duplicated in 3 spellings instead of using SSOT |
| 4 | `tracing` | **5** | 6 | 4 | `phi_init` snapshot timing diverges; missing `assert(s>0)` in Boozer GC; no JAX mu-conservation test; `bracket_root_jax` NaN-poisoning on `fb==fa`; C++ back-fills to `t=tmax`, JAX does not |
| 5 | `regular_grid_interp` | **3** | 3 | 2 | JIT-cache staleness on mutable cell_table; int32 overflow at fine grids; `degree>3` not cross-oracle tested |
| 6 | `dipole_field` | **1** | 2 | 4 | Silent C++↔JAX divergence at dipole-on-axis with cylindrical/toroidal `coordinate_flag`; SIMSOPT-wide `dA_by_dX` pybind docstring contradicts kernel storage |
| 7 | `pm_optimization` | **3** | 3 | 2 | `1e50` sentinel collides with real costs when `‖b‖~1e26` → argmin picks unavailable slot; `nu=0` silent NaN; `nu<0` runs with non-convex Hessian |
| 8 | `wireframe_field` | **2** | 4 | 4 | Non-ISO VLA in C++ `magneticfield_wireframe.cpp:39-40`; unsafe int32 narrowing at ingress |
| 9 | `wireframe_optimization` | **1** | 2 | 2 | RCLS `jnp.linalg.lstsq(rcond=None)` vs scipy: `diff_max ≈ 1.56e14` on rank-deficient LHS (live-verified); uninitialized `int opt_ind_prev` in C++ |
| 10 | `surface_*` | **2** | 3 | 2 | `_scatter_matrix` dense allocation (88MB/call at mpol=ntor=20); `_unitnormal` NaN propagation through jacfwd at near-zero ‖n‖ |
| 11 | `curve_geometry` | **1** | 1 | 3 | Quaternion `|q|=0` classical `0*inf=NaN` autodiff trap (forward correct, gradient NaN where C++ analytic is 0); C++ off-by-one `i<2` instead of `i<3` at `curveplanarfourier.cpp:411` |
| 12 | `boozer_radial_interp` | **3** | 3 | 2 | `B·∇ζ = G(s)` identity tested NOWHERE; `mn_factor` exponent sign inverted in docstring (`s^{-m/2}` not `s^{m/2}`); `enforce_qs=True` and `enforce_vacuum=True` never tested |
| 13 | `analytic_fields` | **0** | 3 | 3 | C++ Dommaschk OMP race pattern (locals declared outside parallel region); `DommaschkJAX.coeffs` mutability vs CPU re-read → silent stale-state divergence |
| **TOTAL** | — | **26** | **38** | **37** | — |

(Counts are NEW second-pass findings only. They are additive to the first-pass tally and the corrigendum.)

---

## Six cross-cutting themes (NEW vs first pass)

### CT-D1 — Autodiff NaN cliffs from `jnp.where`/`safe_*` patterns

Appears in **5 rows**: 2, 3, 7, 10, 11.

The pattern is invariant: `jnp.where(cond, expensive_op(x), fallback)` evaluates `expensive_op(x)` unconditionally, and if that produces NaN/inf the gradient is `0 * NaN = NaN` regardless of which branch is selected. This is the classic JAX trap.

| Row | Site | Pattern |
|----:|------|---------|
| 2 | `integral_bdotn_jax.py:65-78` `"local"` definition | `1/|B|²` where `|B|²=0` returns inf in primal, finite-on-survivors in gradient |
| 3 | `boozer_residual_jax.py` `_inverse_modB` | returns `+inf` at `|B|=0`; gradient is non-finite, untested |
| 7 | `pm_optimization.py:2122-2125` `projection_l2_balls` | First-pass already flagged; `jnp.maximum` propagates NaN, fmax fix needed (already in corrigendum) |
| 10 | `surface_fourier_jax.py:1114-1115`, `surface_rzfourier.py:680-683` `_unitnormal` | `jacfwd` at near-zero ‖n‖ returns `[inf, NaN, NaN]` and all-NaN Jacobian (live-tested at `n=(1e-300, 0, 0)`) |
| 11 | `curveplanarfourier.py:17-22` quaternion normalization | `jnp.where(norm_sq>0, 1/sqrt, 1)` — forward correct at `q=0`, gradient is NaN; C++ analytic returns 0 |

**Recommended audit-level action:** systematic search for `jnp.where(.*0.*, .*x/0.*, ...)` patterns and apply the "double-where" idiom: `safe = jnp.where(cond, x, ones); jnp.where(cond, f(safe), fallback)`. Land a project lint rule.

### CT-D2 — Silent shape/dtype validation gaps

Appears in **5 rows**: 2, 5, 6, 7, 8.

JAX broadcasting and dtype promotion happen silently where C++ would raise. Every row has at least one silent-wrong-answer path.

| Row | Site | Pattern |
|----:|------|---------|
| 2 | `integral_bdotn_jax.py:50` | `target.shape=(5,7)` broadcasts against `B.shape=(1,1,3)` → returns wrong `J`; CPU raises `RuntimeError` |
| 5 | `regular_grid_interp.py:510-525` | `int32` cell-index arithmetic overflows at `nx=ny=nz=1000, degree=4` (`6.4e10 > INT_MAX`) |
| 6 | `dipole_field` validation | JAX raises `ValueError` on `unitnormal.shape != points.shape`; C++ silently accepts (OOB-read potential); both silently accept unknown `coordinate_flag` typos |
| 7 | `pm_optimization` argmin | `jnp.argmin` treats NaN as smallest; no `isfinite` guard at 5 callsites |
| 8 | `wireframe.py:115`, `wireframefield_jax.py:26` | `casting='unsafe'` int64→int32 narrowing with no overflow guard |

**Recommended action:** add a single `_validate_shape_and_dtype` helper at every JAX kernel ingress.

### CT-D3 — Sentinel-value collisions with realistic inputs

Appears in **2 rows** but the pattern is generalizable.

| Row | Site | Sentinel |
|----:|------|----------|
| 6 | `dipole_field.cpp:776-777` | `1e5` distance-filter sentinel breaks for device-distances > 316 m |
| 7 | `pm_optimization.py:609-611, 776-778, 1062-1065, 1597-1600` | `1e50` for unavailable GPMO slots; live-verified to collide with real costs when `‖b‖~1e26` (real costs `~1.27e52 > 1e50`) → argmin returns first sentinel |

**Recommended action:** replace finite sentinels with `jnp.inf`/`std::numeric_limits<double>::infinity()` everywhere. This is the textbook fix and incurs no perf cost on argmin.

### CT-D4 — Stale-state under construction-time capture

Appears in **5 rows**: 5, 7, 8, 12, 13.

JAX modules frequently capture configuration arrays at construction (frozen dataclass spec) but their consumers continue to hold a CPU-side mutable reference. The CPU path re-reads on every call; the JAX path uses the trace-time snapshot.

| Row | Site | Stale-state symptom |
|----:|------|---------------------|
| 5 | `interpolated_field.py:346-381` `_DeviceSpec` cache | mutate `spec.cell_table` → CPU sees new data, JAX returns stale |
| 7 | `pm_optimization` shape recompile | every change in `N` or `P` retriggers ~100ms JAX recompile |
| 8 | `wireframefield_jax.py:60-70, 116-119` | `dB_by_dcoilcurrents` not invalidated on `set_points_*` (latent, currently always overwritten) |
| 12 | `boozermagneticfield_jax.py` | `bri.psi0` mutable; post-construction mutation desyncs JAX K-spline factor |
| 13 | `dommaschk_jax.py:79` `DommaschkJAX.coeffs` | captured in `_spec`; in-place mutation of `self.coeffs` → CPU re-reads, JAX uses snapshot — **silent divergence** |

**Recommended action:** every JAX adapter that exposes mutable CPU attributes should either (a) deep-copy at construction and ban mutation, (b) re-snapshot on every entry, or (c) provide an explicit `invalidate_cache()` method documented in the public API.

### CT-D5 — API surface drift / docstring vs storage contradictions

Appears in **3 rows**: 3, 6, 8.

| Row | Site | Drift |
|----:|------|-------|
| 3 | `boozer_residual_coil_vjp`, `_boozer_residual_vector_composed` | default `weight_inv_modB=False`; sibling APIs default `True` |
| 6 | `python_magneticfield.cpp:45` | SIMSOPT-wide pybind11 docstring claims `dA_by_dX = ∂_j A_l`; actual storage is `dA[p, j, k] = ∂A_j/∂x_k` (transposed for `dA`; for `dB` Hessian symmetry hides it) — load-bearing for downstream consumers |
| 8 | `wireframefield_jax.py:47` | `_simsopt_jax_native_field = True` over-claims: no `B_vjp` implemented → AttributeError for `MagneticFieldSum([WireframeFieldJAX, BiotSavartJAX]).B_vjp(v)` |

### CT-D6 — Identity-level / production-scale test coverage gaps

Appears in **6 rows**: 4, 9, 10, 11, 12, 13.

The first pass found function-by-function parity. The second pass shows that **the physics invariants are tested nowhere**, the production scales are not exercised, and end-to-end identities aren't pinned.

| Row | Missing invariant / scale |
|----:|---------------------------|
| 4 | No JAX-isolated mu-conservation test on any Boozer GC mode; no energy-conservation pin |
| 9 | GSCO parity test uses `nGrid=5, nLoops=2` — way below production |
| 10 | All surface tests use `mpol≤2, ntor≤2`; no high-resolution stress |
| 11 | Production Planar fixture seeds `q ≈ (1, ε, ε, ε)` — near identity, never exercises hard rotation; `gammadashdash`/`gammadashdashdash` not pinned for RZ/Planar/Helical |
| 12 | `B·∇ζ = G(s)` Boozer identity tested in NEITHER item 32 nor item 33; both backends could agree on the same bug |
| 13 | No JAX `trace(dB)=0` Dommaschk; no `B_φ=-1` Reiman direct assertion; no odd-`k` Reiman parity |

---

## Consolidated NEW HIGH findings (second pass, severity-ordered)

### DH1 — Wireframe RCLS rank-deficient LHS divergence (Row 9, live-verified)

`jnp.linalg.lstsq(rcond=None)` vs `scipy.linalg.lstsq` on near-singular LHS produced `diff_max ≈ 1.56e14` on a 6×6 matrix with singular values `[1, 0.5, 0.25, 1e-13, 1e-15, 1e-17]`. First pass categorized as MEDIUM/INFO; live measurement upgrades to HIGH.

**Fix:** explicit `rcond = max(LHS.shape) * jnp.finfo(LHS.dtype).eps` at `wireframe_optimization_jax.py:149`. Add a rank-deficient regression test.

### DH2 — PM optimization sentinel collision (Row 7, live-verified)

`1e50` GPMO sentinel collides with real costs when `‖b‖~1e26`. `argmin` returns first sentinel slot. Affects all 5 GPMO variants.

**Fix:** replace `1e50` with `jnp.inf` at 5 sites (`py:609-611, 776-778, 1062-1065, 1597-1600`).

### DH3 — PM optimization `nu=0` silent NaN (Row 7, live-verified)

`nu=0` produces `1/(2·0)=inf` in `_hessian_action:2271` and `m_proxy/0` in `ATb_rs:2464`. No validator.

**Fix:** add `nu > 0` validator at API ingress; document `nu=0` as forbidden.

### DH4 — Tracing `phi_init` snapshot timing diverges (Row 4)

C++ `ToroidalTransitStoppingCriterion` captures `phi_init` after the first integration step (`iter==1` in `tracing.h:30-44`); JAX captures from `y0` at driver-construction time (`tracing.py:886, 1422, 2882`). Produces a small but compounding offset.

**Fix:** capture `phi_init` after the first JAX step inside the criterion predicate.

### DH5 — Tracing missing `assert(s>0)` for Boozer GC RHS (Row 4)

`assert(ys[0]>0)` at `tracing.cpp:226` has no JAX equivalent. Boozer particles can silently spiral past the axis with NaN derivatives; only the Cartesian `dtmax` discipline in C++ keeps the RHS away.

**Fix:** add `jnp.where(s > 0, rhs, NaN)` and surface to driver as `status=-2`.

### DH6 — Tracing no JAX-isolated mu/energy conservation test (Row 4)

`dv_par = -(mu/v_par)·Σ ∂B/∂q · dq/dt` form is fragile to wrong-sign bugs; only endpoint parity is tested.

**Fix:** add `tests/jax_core/test_tracing_jax_conservation.py` with assertions `|μ_t − μ_0| < 1e-10` and `|E_t − E_0| < 1e-10` over `t > T_bounce` for trapped orbits.

### DH7 — Tracing `bracket_root_jax` NaN-poisoning (Row 4)

`tracing.py:765` computes `candidate = b - fb * width / (fb - fa)` unconditionally; when `fb == fa` this is NaN and poisons subsequent iterations via the `sign(fa)*sign(fc) <= 0` keep-left rule that treats NaN as False.

**Fix:** `jnp.where(jnp.abs(fb-fa) > 1e-300, false_position, bisection_midpoint)`.

### DH8 — Tracing C++ back-fills to t=tmax; JAX does not (Row 4)

C++ `dense.calc_state(tmax, y)` at `tracing.cpp:441-444` ensures `loss_ctr` accounting at field/tracing.py is consistent. JAX does not back-fill, over-counting losses at the same RNG seed. First pass marked PARITY; second pass shows structural drift.

**Fix:** add explicit end-of-trajectory step to land on exactly `t=tmax` for non-terminated particles.

### DH9 — Integral B·n silent shape broadcast (Row 2, live-verified)

`integral_bdotn_jax.py:50` silently broadcasts `target.shape=(5,7)` against `B.shape=(1,1,3)` and returns `J=17.5` (totally wrong); CPU raises `RuntimeError`. **No shape validation at all in the JAX kernel.**

**Fix:** add `jnp.shape` assertion at kernel ingress; add a wrong-shape regression test.

### DH10 — Integral B·n `"local"` definition silent inf-gradient (Row 2, live-verified)

`"local"` with one `|B|=0` quadrature point returns `J=+inf` but a **finite** gradient on the other points. CPU `SquaredFlux.dJ()` raises typed `ObjectiveFailure`. The `SquaredFluxJAX` adapter does catch this via `value=inf` but bare kernel + any caller that doesn't co-emit value+grad slips through.

**Fix:** add explicit guard `jnp.where(any_zero_modB, NaN, grad)` to make the gradient path also surface the failure.

### DH11 — Integral B·n surface-DOF drift guard untested (Row 2)

`fluxobjective_jax.py:378-386` `_raise_if_surface_dofs_drifted` has zero test coverage (verified by grep).

**Fix:** add regression test that constructs `SquaredFluxJAX`, mutates underlying surface DOFs, and asserts the guard fires.

### DH12 — Boozer residual `_split_decision_vector` no bounds check (Row 3)

`boozer_residual_jax.py:73-82` does not validate `surface_size`; negative values silently produce wrong shapes via `jnp.take`. Production callers route through `_pack_optimizer_state` so it doesn't currently bite, but a future caller with `x.shape == (1,)` and `optimize_G=True` silently produces `(empty_sdofs, x[-1], x[0])`.

**Fix:** add `surface_size >= 0` assertion at function ingress.

### DH13 — Boozer G-from-currents formula duplicated in 3 spellings (Row 3)

`_unpack_decision_vector` (`boozer_residual_jax.py:554-556`) duplicates `G = μ₀·Σ|I_k|` inline instead of calling SSOT `compute_G_from_currents` (`label_constraints_jax.py:49-61`). Three different spellings exist: `4.0e-7*np.pi`, `4.0*jnp.pi*1e-7`, and `2*np.pi*…*(4*np.pi*1e-7/(2*np.pi))` in C++ oracle. Currently algebraically identical, but no test guards against drift.

**Fix:** delete the inline duplicate and import from SSOT.

### DH14 — Regular-grid JIT-cache staleness on `cell_table` mutation (Row 5)

`frozen=True` dataclass with mutable NumPy fields; `evaluate_batch` re-stages each call but `_DeviceSpec` (`interpolated_field.py:346-381`) caches once. Two consumer paths over one spec disagree silently when underlying numpy data is mutated.

**Fix:** make `_DeviceSpec` data id keyed (cache invalidation on numpy array id change), or deep-copy on construction.

### DH15 — Regular-grid int32 overflow at fine grids (Row 5)

`regular_grid_interp.py:510-512, 525` flat cell-index arithmetic and `idx_dof` at `regular_grid_interpolant_3d.h:124-127` use int32. UB in C++ at `nx=ny=nz=1000, degree=4` (`6.4e10 > INT_MAX`).

**Fix:** promote to int64 in both backends.

### DH16 — Regular-grid `degree>3` cross-oracle gap (Row 5)

Wrapper test pins `_DEGREE=4` but truncation residual O(1e-9) hides cell-level einsum-vs-FMA O(1e-13) divergence. Bare-kernel `test_cpp_cross_oracle` covers only `degree in [1,2,3]`.

**Fix:** add explicit `degree=4` cross-oracle test on a polynomially-exact fixture (`nx·degree` knots interpolating a degree-`(nx·degree-1)` polynomial — residual = 0 in exact arithmetic).

### DH17 — Dipole-on-axis with cylindrical coord_flag silent divergence (Row 6, live-verified)

Dipole at `(0,0,0)` with `coordinate_flag="cylindrical"` or `"toroidal"`: **C++ returns `[NaN, NaN, finite-z]` while JAX returns finite values**. Root cause: `xsimd::atan2(0, 0)` returns NaN; `jnp.atan2(0, 0)` returns 0 (matches `std::atan2`). C++ at `dipole_field.cpp:333-334` propagates NaN through rotation factors `cphi_new`/`sphi_new` and poisons the output A-matrix.

**Fix:** either guard `atan2(0, 0)` with a sentinel in C++ to match JAX (recommended), or document that dipoles must not lie on the axis for non-cartesian frames.

### DH18 — Surface `_scatter_matrix` dense allocation (Row 10)

`surface_rzfourier.py:327-337` builds dense `(target × source)` float64 matrices per call. At `mpol=ntor=20` this is ~88MB host→device transfer per `_coefficients_from_dofs` invocation, called inside every `jacfwd`/`vjp`. The XYZ-tensor path uses `lax.scatter` (`surface_fourier_jax.py:1203-1234`) — inconsistent.

**Fix:** replace dense scatter with `lax.scatter` to match the XYZ-tensor path.

### DH19 — Surface `_unitnormal` NaN propagation through jacfwd (Row 10, live-verified)

`surface_fourier_jax.py:1114-1115`, `surface_rzfourier.py:680-683`. At `n=(1e-300, 0, 0)`, `jacfwd` returns `[inf, NaN, NaN]` and all-NaN Jacobian. Existing degenerate-surface test only checks exact zero, missing near-zero.

**Fix:** double-where pattern around the normalization.

### DH20 — Curve quaternion `|q|=0` autodiff NaN trap (Row 11)

`curveplanarfourier.py:17-22`. Classic JAX `0*inf=NaN` trap. Forward value at `q=0` matches C++ (identity rotation), but `jax.grad` w.r.t. quaternion DOFs at `q=0` returns NaN where C++ analytic derivative returns 0. Existing test `tests/geo/test_curve.py:1186-1265` seeds `q=(0,0,0,0)` and only checks forward.

**Fix:** double-where pattern; add gradient parity test at `q=0`.

### DH21 — Boozer radial: `B·∇ζ = G(s)` identity tested nowhere (Row 12)

First-pass audit punted to item 33 wrapper; item 33 test file (239 lines) only does JAX-vs-CPU re-evaluation, never closes the identity. **Both backends could agree on the same bug.**

**Fix:** add an identity-level assertion: compute `B·∇ζ` via JAX wrapper, compare against the precomputed `G(s)` spline coefficients within tolerance.

### DH22 — Boozer radial `mn_factor` exponent sign (Row 12)

First-pass brief and the JAX module docstring describe `mn_factor = s^{|m|/2}`; the actual code uses `s^{-m/2}` (negative exponent). The actual axis-singularity behavior is governed by polynomial EXTRAPOLATION of `mn_factor` outside `[s_half_mn[0], 1]`, which is finite but physically incorrect outside the validity range. Untested regime.

**Fix:** clarify documentation; add test at `s < s_half_mn[0]` to pin the extrapolation behavior.

### DH23 — C++ wireframe non-ISO VLA / stack-overflow risk (Row 8)

`magneticfield_wireframe.cpp:39-40` declares `double* halfPrd_ptr[nHalfPrds]; double seg_signs[nHalfPrds];` with `nHalfPrds` runtime-determined. Non-ISO (variable-length arrays are not standard C++), MSVC-unfriendly, stack-overflow vector for pathological `nHalfPrds`.

**Fix:** convert to `std::vector` or fixed-size static buffer with assertion on `nHalfPrds ≤ MAX`.

### DH24 — C++ wireframe optimization uninitialized `int opt_ind_prev` (Row 9)

`wireframe_optimization.cpp:154`. Declared uninitialized. Safe today only because the synchronized `break` after `stop_none_eligible=true` prevents the gated read at L270 from ever being reached without prior assignment. A future maintainer who replaces `break` with `continue` invokes UB.

**Fix:** initialize to `-1`; JAX already does this correctly at `wireframe_optimization_jax.py:473`.

### DH25 — Wireframe unsafe int32 narrowing (Row 8)

`wireframe.py:115`, `wireframefield_jax.py:26` use `casting='unsafe'` int64→int32 cast with no overflow guard. `ToroidalWireframe.segments` is int64.

**Fix:** add an explicit overflow check; raise on `n_segments > INT32_MAX`.

### DH26 — Wireframe `_simsopt_jax_native_field` over-claims (Row 8)

`wireframefield_jax.py:47` sets `_simsopt_jax_native_field = True` but no `B_vjp` is implemented. A `MagneticFieldSum([WireframeFieldJAX, BiotSavartJAX]).B_vjp(v)` would AttributeError.

**Fix:** either implement `B_vjp` or set the flag to `False`.

---

## Updated fix list (combining first-pass + corrigendum + deeper)

### P0 — Correctness fixes (must ship)

- **F1 (first-pass H7 + corrigendum):** `projection_l2_balls` NaN fix at `pm_optimization.py:2123` — both `ones_like` AND `fmax`.
- **F2 (first-pass H1 → MEDIUM-doc):** Boozer residual docstring at `boozer_residual_jax.py:31-34`.
- **F-DH2 (deeper DH2):** Replace `1e50` GPMO sentinel with `jnp.inf` at 5 sites.
- **F-DH3 (deeper DH3):** Add `nu > 0` validator at PM optimization API ingress.
- **F-DH9 (deeper DH9):** Add shape validation at `integral_bdotn_jax.py:50` ingress.
- **F-DH1 (deeper DH1):** Add explicit `rcond` to `jnp.linalg.lstsq` in RCLS at `wireframe_optimization_jax.py:149`.

### P1 — Tracing parity hardening

- F3 (5-site `dtmax` clamp), F4 (per-mode initial step), F5 (atol fix) — from first pass.
- **F-DH4 (DH4):** capture `phi_init` after first JAX step.
- **F-DH5 (DH5):** add `assert(s>0)` equivalent in JAX Boozer GC.
- **F-DH6 (DH6):** add mu/E conservation test.
- **F-DH7 (DH7):** fix `bracket_root_jax` NaN poisoning.
- **F-DH8 (DH8):** add `t=tmax` back-fill in JAX driver.

### P2 — Interp boundary

- F6 (`floor → trunc`), F7 (API decision required for `leave-unchanged`) — first pass.
- **F-DH14 (DH14):** `_DeviceSpec` cache invalidation.
- **F-DH15 (DH15):** promote int32 → int64 for flat cell indices.
- **F-DH16 (DH16):** add `degree=4` cross-oracle test.

### P3 — Upstream coordinated

- F9 (GSCO operator precedence), F10 (boozer radial OMP `reduction` fix using scalar accumulators) — corrected from corrigendum.
- **F-DH23 (DH23):** replace C++ VLA with `std::vector` in wireframe.
- **F-DH24 (DH24):** initialize `opt_ind_prev` in wireframe optimization C++.

### P4 — Autodiff NaN cliffs (NEW priority tier from CT-D1)

Apply the **double-where pattern** at:
- **F-DH10 (DH10):** `integral_bdotn_jax.py:65-78` `"local"` definition gradient guard.
- **F-DH19 (DH19):** `_unitnormal` in `surface_fourier_jax.py:1114-1115` and `surface_rzfourier.py:680-683`.
- **F-DH20 (DH20):** quaternion normalization at `curveplanarfourier.py:17-22`.
- **DH:** `_inverse_modB` gradient at |B|=0 in `boozer_residual_jax.py`.

### P5 — Stale-state contracts (NEW priority tier from CT-D4)

For every JAX adapter:
- Document mutation policy.
- Either deep-copy at construction, re-snapshot per entry, or provide `invalidate_cache()`.
- Specific sites: F-DH14 (cell_table), `bri.psi0` (Row 12), `DommaschkJAX.coeffs` (Row 13), `dB_by_dcoilcurrents` (Row 8).

### P6 — Test oracle hardening

(All first-pass F11-F18 items remain, plus:)
- **F-DH11 (DH11):** surface-DOF drift guard regression test.
- **F-DH12 (DH12):** `_split_decision_vector` bounds check.
- **F-DH13 (DH13):** delete the duplicated G-from-currents formula.
- **F-DH21 (DH21):** Boozer `B·∇ζ = G(s)` identity test.
- **F-DH22 (DH22):** `mn_factor` extrapolation test.

### P7 — Documentation / API surface

- F21 (stale tracing docstring), F22 (JAX-only exports) — first pass.
- **F-DH26 (DH26):** fix `_simsopt_jax_native_field` over-claim.
- **F-DH:** Row 6 — fix SIMSOPT-wide `dA_by_dX` pybind docstring at `python_magneticfield.cpp:45`.
- **F-DH:** Row 3 — reconcile default `weight_inv_modB` kwarg across sibling APIs.

---

## Revised effort roll-up

| Tier | Items | Effort |
|------|-------|--------|
| P0 (correctness, must-ship) | F1, F2, F-DH2/3/9/1 | ~5-6 hours |
| P1 (tracing parity) | F3-F5, F-DH4/5/6/7/8 | ~3 days |
| P2 (interp boundary) | F6, F7 (decision), F-DH14/15/16 | ~2 days |
| P3 (upstream coordinated) | F9, F10, F-DH23/24 | ~1.5 days |
| P4 (autodiff NaN cliffs, NEW) | F-DH10/19/20 + inverse_modB | ~1 day |
| P5 (stale-state contracts, NEW) | mutation-policy across 5 adapters | ~2 days |
| P6 (test oracles) | F11-F18 + F-DH11/12/13/21/22 | ~7 days |
| P7 (docs/API) | F21, F22, F-DH26, dA docstring, kwarg reconciliation | ~6 hours |
| **TOTAL** | 45+ items | **~3 weeks working time** |

---

## What's left to find

This audit has now spent ~10 hours of agent time across 24 subagent passes (12 first-pass + 12 deeper-pass). The convergence of finding rates suggests the remaining novel-finding curve is flattening for forward-formula and code-level issues. The unexplored areas are:

1. **Integration-level**: behavior of these modules under composition. Stage-1+Stage-2 outer loops; single-stage optimization with all modules engaged.
2. **GPU-specific**: parity behavior under CUDA. Reduction-order, atomicAdd, FP determinism.
3. **Performance at production scale**: production stellarators have `nphi≈64, ntheta≈64, mpol≈10, ntor≈10, ncoils≈5-50, npoints≈10^5`. No audit covered scale.
4. **Cross-platform**: macOS (Apple Silicon) vs Linux x86_64 vs GPU; the LS path is documented as non-portable.
5. **Long-trajectory robustness**: tracing over `tmax ≈ 1 sec` (~10^9 RHS evaluations); only short runs are tested.
6. **Concurrent-call safety**: behavior under `multiprocessing` / `dask` parallelism.

A third pass focusing on these would be productive but is qualitatively different from the first two (more integration testing, less static code review).

---

## Confidence assessment

- **Forward formula parity (first pass):** very high. Formulas, signs, factors, normalizations cross-verified.
- **Code-level correctness (deeper pass):** medium-high. Several latent bugs found; convergence is flattening.
- **Production safety:** still uncertain. The autodiff NaN cliffs (CT-D1) and silent validation gaps (CT-D2) are the highest residual risk. The fix list above closes the known ones; an integration-level pass would close the remaining unknowns.
- **Cross-platform / GPU determinism:** not assessed. The project's own GPU smoke tests are the right place; this audit did not exercise them.
