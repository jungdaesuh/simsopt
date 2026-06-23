# JAX Port Performance/Memory Optimization Implementation Plan

## Purpose

Track the concrete remediation of the performance/memory optimization opportunities found by the
multi-agent JAX audit (9 subsystem finders -> 51 candidates -> adversarial verification). The original
write-up contained 16 named findings because #3 was dropped from the ranked list; this plan restores #3,
ranks the missed adjacent sites found during validation, and was doc-review revalidated against live source
at HEAD `5fe184308`.

This file is the single execution + review + progress-tracking artifact for that work. Each task carries
its file:line anchor, the concrete JAX-level change, the parity guard, and a runnable validation command so
another engineer/agent can execute it without re-deriving the audit.

## Goals

- Eliminate the two confirmed Tier-1 hot-path inefficiencies on the single-stage Newton/LBFGS plumbing
  (closure retrace; un-fused host Hessian build) with **no** numerical-parity regression.
- Land the bit-exact mechanical wins (tracer-safe curve slice, scan-invariant hoists, scatter writes,
  reduction micro-fixes) that are pure FLOP/allocation reductions.
- Reduce XLA graph-breadth / compile pressure in the Stage-2 ALM build and the Cartesian/cylindrical
  interpolation tracing path (including the second un-flagged caller the original audit missed).
- Right-size the secondary-solver (pm/wireframe, self-field-force) materializations that are real but
  off the single-stage path, without disturbing C++ bit-parity tests.
- Keep every change reviewable as an isolated diff with a green targeted test.

## Non-Goals

- No changes to production physics kernels on the single-stage path (BiotSavart B/dB, surface-Fourier eval,
  Boozer residual objective/VJP/HVP). A1 is explicitly a low-priority public/test dense-Jacobian helper, not
  a production Boozer residual rewrite.
- No re-work of already-resolved poles: the dense LS-Hessian adjoint assembler is already chunked
  (`lax.map` + `SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE` at `optimizer.py:3605,3642`), and dense
  finalization is already byte-gated by the live `max_dense_hessian_bytes` / `max_dense_jacobian_bytes`
  policy surfaces. The outer LBFGS loop is already host-driven.
- No blanket float32 downcast — this is an FP64-critical codebase.
- No action on the guarded watch-item (`solve/dispatch.py:539`) unless a production route is found that
  enables dense linearization without a byte cap.
- Not touching `src/simsopt/field/force.py` (original simsopt, out of scope — "jax-ported code only").

## Current Context

- Scope: `src/simsopt_jax/` + `src/simsopt_jax_adapters/` (~62k LOC). JAX 0.10.0; `nvidia-smi` host must be
  ≥R575 to link 0.10's 12.9 cubins (GPU runs only).
- Single-stage hot path = the per-evaluation Newton/LBFGS plumbing; it is the GPU budget center of gravity
  and came out essentially clean — only Phase 1 items land on it.
- Parity convention: bit-exact is preferred; the existing chunked-reduction order change of ~1e-16 is an
  already-accepted tolerance (documented at `optimizer.py:3636-3637`), well below the 1e-11 Newton tol.
- Test runner: targeted `pytest` files are listed per task. This plan is source-validated but the current
  doc-review pass did **not** rerun local JAX kernels/tests; execute the listed commands under the repo's
  configured Python/JAX environment before checking off implementation tasks. For CPU parity checks, use
  `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python3 -m pytest <file>`. The suite is **not** xdist-safe and the JAX
  cache can accumulate across tests — run per-file (or rely on the conftest per-test `jax.clear_caches`).

## Rationale

Order of attack = leverage × (1 / parity-risk). Phase 1 first: highest leverage (eliminates up to
`max_outer−1` full inner-solve recompiles; restores the peak-memory bound + HVP fusion on the Newton-polish
path) at negligible/low parity risk, and both are localized. Phase 2 next: pure bit-exact wins that are
safe to batch. Phase 3: structural/compile-breadth changes that need pair-set or signature bookkeeping.
Phase 4: secondary solvers and diagnostics where payoff is real but smaller and the main risk is disturbing
C++ bit-parity oracles. The watch-item is intentionally deferred.

## Assumptions

- A change that only reorders an FP64 reduction (≤ ~1e-16 rel) is acceptable where a bit-exact form is not
  cheaply available; bit-exact is used wherever the slice/scatter/hoist is an exact reindex.
- The Stage-2 batched kernel `pairwise_selected_smoothmin_distance_batched_pure` is proven equal to the
  pure form at ~1e-12 (`tests/geo/test_surface_objectives_jax.py`), so routing through it preserves values.
- `_evaluate_batch_jit` already exposes `strict_cell_order` (static arg, default `True`,
  `regular_grid_interp.py:693/722`); only the cyl/cart wrappers and `surface_classifier` fail to thread it.
- pm/wireframe items are secondary solvers (not single-stage); their tests gate C++ bit-parity, so the
  default behavior must be preserved (opt-in flags only).

## Implementation Plan

### Phase 1 — Tier 1, single-stage hot path (do first; parity-safe)

1. **T1.2 (#14) — Augmented-Lagrangian closure retrace** · `src/simsopt_jax/solve/serial.py:381-418`
   - [ ] Move `augmented_objective` definition **out** of the `for _outer_index in range(...)` loop;
         define it **once** as `augmented_objective(candidate_x, _args)` and unpack
         `multipliers, penalty_weight = _args` inside it.
   - [ ] Pass `args=(current_multipliers, current_penalty_weight)` to `optx.minimise`/`least_squares`
         instead of capturing the freshly-reassigned tracers from the loop body (reassigned at ~`:415,:418`).
   - [ ] Keep the two `jax.debug.callback` loggers captured (they are loop-invariant) and keep
         `penalty_weight` a 0-d `jnp` array of stable dtype/shape so the trace signature is constant.
   - [ ] Confirm the inner solve compiles once: set `JAX_LOG_COMPILES=1` and assert 1 compile after the
         first outer iteration (was +1 per outer iteration).

2. **T1.1 (#1) — Host dense-Hessian un-fused Python column loop** · `src/simsopt_jax/geo/optimizers/optimizer.py:3646-3654`
   - [ ] Delete `_materialize_dense_linear_operator_host` (the `for index in range(...)` + `jnp.stack`
         selector build).
   - [ ] Route the **Hessian** host build to the already-chunked, **symmetrized** device sibling:
         `_materialize_dense_hessian_host` (defined `:3687`, wraps the host linear op `:3646`) should delegate
         to `_materialize_dense_hessian` (`:3679`), which wraps the chunked `_materialize_dense_linear_operator`
         (`:3632`, `lax.map(..., batch_size=_DENSE_OPERATOR_CHUNK_BATCH_SIZE)`, constant `:3605`). Delegate to the
         symmetrized Hessian wrapper, **not** the bare linear operator, so symmetrization is preserved. The
         host-jax lane selector is `materialize_dense_hessian_fn = _materialize_dense_hessian_host if allow_host_control else _materialize_dense_hessian` (`:4871-4875`).
   - [ ] Add a host-vs-device build agreement assertion (~1e-12 rel-L2) so the reduction-order delta is bounded.

### Phase 2 — Bit-exact / mechanical wins (batchable; pure FLOP/allocation reductions)

3. **T3.1 (#7, refined) — Curve DOF slice via dense one-hot selector under jit** · `src/simsopt_jax/core/curve_geometry.py:129-176`
   - [ ] Replace the **tracer** branches (`_slice_1d_static_selector` / `_update_1d_static_selector`, which
         actually run under jit) with a **tracer-safe static slice**: `lax.slice_in_dim` (static int bounds)
         for `_slice_1d_static`, and `slice_in_dim` head/tail + `concatenate` for `_update_1d_static`.
   - [ ] Before deleting the helpers, add a durable regression/guard probe in the test suite for
         `jax.jit` + `transfer_guard("disallow")` covering static-bound `lax.slice_in_dim` in both value and
         jitted-gradient form. Earlier scratch probes were not present in the repo during this doc review, so
         this plan must not treat the transfer-guard question as closed until the executable test exists.
   - [ ] Delete the two selector helpers and replace both branches unconditionally with
         `lax.slice_in_dim` (slice) / `slice_in_dim` head+tail + `concatenate` (update).

4. **T2.2 (#9) — GPMO scan-invariant `col_sq`/`penalty` recompute** · `src/simsopt_jax/core/pm_optimization.py:793-806` (+ `986`, `1532`, `2133`)
   - [ ] Hoist `penalty = reg_l2 * _component_mmax(m_maxima)**2` (`:793`) and `col_sq = sum(A*A, axis=0)` (`:796`)
         **before** the greedy scan; pass as optional precomputed args (mirror how `gpmo_arbvec_solve` hoists
         `contributions` at `:1149`), computing them only when `None`.
   - [ ] Thread the precomputed pair through both `gpmo_baseline_solve` scan branches and the backtracking
         path (`:2133`). Do **not** attempt the rank-1 Gram update.

5. **T2.6 (#13) — GSCO full-vector `jnp.where` history rewrite** · `src/simsopt_jax/core/wireframe_workflow.py:219-223`
   - [ ] Replace `_update_vector_entry`'s `jnp.where(positions==index, ...)` (O(capacity)/record) with
         `vector.at[index].set(value)`.
   - [ ] Fold the outer `should_record` gate into the index via an out-of-bounds sentinel + `mode="drop"`
         so the redundant outer full-buffer select is removed. Apply to the final-flush updates too.

6. **T3.2 (#15) — `pairwise_sum_axis` transpose+pad on a size-3 axis** · `src/simsopt_jax/core/reductions.py:43-61`
   - [ ] For `axis_size <= 4`, build the identical left-leaning binary tree directly via slices
         (`(s0+s1)+s2`) instead of `moveaxis` + `_pad_axis(_next_power_of_two)` + `_pairwise_reduce_axis0`.
   - [ ] Keep the existing moveaxis/pad path for larger axes. Do **not** widen the `<=4` bound (parity of
         the addition tree must stay identical to the current reduction order).

7. **M1 — Dense LM/Newton full-identity allocations** · `src/simsopt_jax/geo/optimizers/optimizer.py:2435`, `3786` (+ `3125`)
   - [ ] In `_dense_lm_propose_step` (`:2435`, `@jax.jit` at `:2429`) and `_stabilize_dense_hessian` (`:3786`,
         multiplier is `stab_value = _optimizer_scalar(stab, ...)` from `:3785`, not bare `stab`), replace
         `H + damping * jnp.eye(n)` with a diagonal add: `H.at[jnp.diag_indices(n)].add(damping)` (traces fine
         under jit; `_stabilize_dense_hessian`/`_qr_lm_step` are not individually `@jax.jit` but run under a
         jitted caller).
   - [ ] Leave `_qr_lm_step` (`:3125`) as-is unless profiled: its `damping_sqrt * jnp.eye(cols)` block is
         consumed by a dense QR of the augmented Jacobian (structurally needed) — lowest reward of the three.

8. **M2 — Per-cell basis identity** · `src/simsopt_jax/core/regular_grid_interp.py:655`
   - [ ] Only if this file is already open for Phase 3 / T2.5: the `jnp.eye(degree+1)` inside `_basis_values`
         is a tiny compile-time constant; optional micro-cleanup (e.g. precompute the off-diagonal mask once).
         Do not prioritize on its own.

### Phase 3 — Compile-breadth / structural (signature + pair-set bookkeeping)

9. **T2.5 (#4, widened) — Strict fori-loop contraction on the Cartesian/cyl tracing path** · `src/simsopt_jax/core/regular_grid_interp.py:665-682,722` ; `src/simsopt_jax/core/interpolated_field.py:253-261,313,347-355` ; `src/simsopt_jax/core/surface_classifier.py:94`
   - [ ] Add `strict_cell_order: bool` to `_evaluate_cyl_field_jit` and `_evaluate_cart_field_jit`
         signatures **and** their `static_argnames` (currently only
         `nfp,stellsym,unfold_kind,degree,value_size,out_of_bounds_ok`).
   - [ ] Forward `strict_cell_order` at the `_evaluate_batch_jit` call site (`interpolated_field.py:313`,
         and the cart analogue) — it currently defaults to the slow `True`.
   - [ ] Route the second un-flagged caller `surface_classifier.py:94` (`evaluate_batch_device(...)`, default
         `True`) to pass `strict_cell_order=False`.
   - [ ] Default the Cartesian/cyl + classifier tracing paths to the fast `_fused_tensor_contract` einsum
         (`strict_cell_order=False`); keep strict reachable behind a parity-debug flag. The Boozer path
         already opts in (`interpolated_boozer_field.py:757,769`).
   - [ ] Spot-check one Poincaré/DOPRI5 trajectory: long chaotic integrations can shadow-diverge from a
         ~1e-12 FP reassociation even when per-eval parity holds.

10. **T2.1 (#16) — Stage-2 ALM constraints Python-unroll a pair list** · `src/simsopt_jax_adapters/objectives/stage2_target.py:1056-1119`
    - [ ] Replace the nested Python `point_pairs` build feeding `pairwise_selected_smoothmin_distance_pure`
          with the batched `pairwise_selected_smoothmin_distance_batched_pure` (`_pairwise_reductions.py:531`,
          used at `surface_objectives.py:1557,1578`), grouped into homogeneous batches (curve-curve dynamic,
          curve-curve fixed-TF, curve-surface) that **share one global** `hard_min`/`cutoff`/`sum_exp`.
    - [ ] Keep fixed TF-vs-TF pairs as **constant batched arrays** — they cannot be precomputed to a scalar
          because they couple to the dynamic pairs' global hard_min.
    - [ ] Preserve the exact pair **set** (strict lower triangle for self-pairs) — assert pair count
          unchanged (~190 at 20 TF coils).

11. **#11 (now ranked) — RCLS diagonal regularizer materialized + dense matmul** · `src/simsopt_jax_adapters/solve/wireframe.py:99-118,151,155`
    - [ ] For scalar/vector `W`, skip building the full `n×n` `Wmat` and replace the dominant
          `WQ2mat = Wmat @ Q2mat` / `WQ1mat = Wmat @ Q1mat` (`:151,:155`) with row-scaling
          (`W_arr * Q` for scalar, `W_arr[:,None] * Q` for vector).
    - [ ] Keep the genuine dense matmul only for the matrix-form `W` branch (`:111-117`).

### Phase 4 — Secondary solvers & diagnostics (real but off single-stage; guard C++ parity)

12. **T2.3 (#10) — Wireframe normal-field matrix host loop + full device pull** · `src/simsopt_jax_adapters/field/wireframe.py:149-174`
    - [ ] Replace the `for i in range(self._n_segments): matrix[:,i]=...` (`:171-173`) + the full host pull
          with one fused on-device `jnp.einsum('snc,pc->ps', contributions, unitn_flat) * fac_flat[:,None]`,
          copying only the final `(n_points, n_seg)` result to host.
    - [ ] Reshape `unitn`→`(n_points,3)`, flatten `fac`→`(n_points,)`. Preserve the `_dB_by_dcoilcurrents`
          side-effect populated by the sibling method (`:145-147`).

13. **T2.4 (#12) — MwPGP per-iteration diagnostic extra matvec** · `src/simsopt_jax/core/pm_optimization.py:3186-3191`
    - [ ] Add a static `record_residual` flag; route **only** the relax-and-split path through
          `record_residual=False` (it computes convergence independently). The no-L0/no-L1 path must keep
          `record_residual=True` (reports `residual_history[-1]`).
    - [ ] Do **not** reconstruct the proxy from `g_new + ATb_rs` (breaks C++ bit-parity).

14. **T3.3 (#5) — `theta_vmec` dense diagonal Jacobian** · `src/simsopt_jax/core/vmec_fieldlines.py:40-52`; `src/simsopt_jax/core/_root.py:21-25,121-124`
    - [ ] Stop materializing `jnp.diag(diagonal)` for the VMEC theta solve. Thread the diagonal vector through
          a diagonal-specific Newton step (`step = -residual / diagonal`) and implicit backward solve
          (`cotangent / diagonal`) so the solver avoids dense `Ntheta x Ntheta` allocation and LU.
    - [ ] Keep the generic `_root.py` dense path for non-diagonal callers; do not widen this into a public
          root-solver API change unless a caller inventory proves it is needed.
    - [ ] Gate against CPU `vmec_fieldlines` and current JAX results at the existing `test_vmec_fieldlines_jax`
          tolerances; this is not a bit-exact promise because it changes a linear-solve route.

15. **T3.4 (#17) — Not-a-knot spline dense `(3M)^2` solve** · `src/simsopt_jax/core/mhd_bootstrap.py:15-90`
    - [ ] Replace `_not_a_knot_coefficients`' dense assembled `(3*intervals, 3*intervals)` system with a
          compact banded/tridiagonal not-a-knot solve for the cubic coefficients. Keep the dense version only
          as a test oracle if useful.
    - [ ] Preserve the current unit-spacing contract and derivative conventions in `_eval_cubic`; do not
          switch interpolation families.
    - [ ] Gate with `tests/mhd/test_bootstrap_jax.py`; accept tolerance-matched JVP/FD parity, not byte identity.

16. **T3.5 (#6) — VMEC geometry pre-broadcasted mode tensors** · `src/simsopt_jax/core/vmec_geometry.py:390-413`
    - [ ] Avoid materializing the full family of `mcosangle`, `ncosangle`, `mncosangle`, `m2cosangle`,
          `n2cosangle`, `msinangle`, `nsinangle`, `mnsinangle`, `m2sinangle`, and `n2sinangle` tensors.
          Compute weighted mode sums from `cosangle`/`sinangle` with the mode multipliers folded into the
          coefficient argument at each use.
    - [ ] Keep formulas source-local in `vmec_geometry.py`; do not introduce a second VMEC mode-sum SSOT.
    - [ ] Gate with VMEC compute-geometry diagnostics because this touches many derived geometry fields.

17. **T3.6 (#8 / #2) — GSCO candidate gather + coil-coil inductance rank-5 broadcast**
    - [ ] `wireframe_workflow.py:339-340`: hoist the loop-static `Acol` gather into params; keep the direct
          `(r+d)²` form (do **not** expand — catastrophic cancellation).
    - [ ] `field/force.py:1097-1130` `_coil_coil_inductances_pure`: drop the ×3 `r_ij` tensor (preserve the
          exact per-component `+eps`) and `lax.map` over the coil axis. (Off single-stage; advanced
          coil-force example only.)

18. **#3 (dropped finding) — Self-field-force B vmap** · `src/simsopt_jax_adapters/field/force.py:134-155` `_B_at_point_from_coil_set_pure` (def `:105`)
    - [ ] Note: `_B_at_point_from_coil_set_pure` is a **single-point** evaluator; its internal
          `vmap(from_j)(jnp.arange(n))` (`:154`) maps **over coils** (`n` = number of coils), not over
          evaluation points. The unchunked **per-point** mapping happens at the **caller**
          (`_B_field_pure` / `_mutual_B_field_at_point_pure`, which `vmap` this single-point fn over points).
    - [ ] Chunk the per-point `vmap` at that caller with `lax.map` (batch-size knob), and/or `lax.map` the
          coil axis inside `:154` if the coil count is large. Small impact, off single-stage; do only if
          `force.py` is already open for T3.6/A3/A4.

19. **A3 — NetFluxes rank-5 quadrature broadcast** · `src/simsopt_jax_adapters/field/force.py:1397` (def), `:1466-1469` (broadcast) `_net_fluxes_pure`
    - [ ] Replace the `(m,n,m',n',3)` broadcast `gammas_targets[:,:,None,None,:] - gammas_sources[None,None,:,:,:]`
          with a `lax.map`/chunked reduction over the target-coil axis (mirror the T3.6 inductance fix).
          Off single-stage (NetFluxes/induced-current workflow).

20. **A4 — Explicit `L⁻¹` then `@ flux`** · `src/simsopt_jax_adapters/field/force.py:1195-1202` (inverse), `:1255-1264` (`@ _net_fluxes_pure` token at `:1257`) `_coil_coil_inductances_inv_pure` / `_induced_currents_pure`
    - [ ] Replace forming `inv_L` (two `_solve_triangular_columns` against a full `m×m` identity) + matmul
          with a direct Cholesky solve of `L·I = −flux` against the `m`-vector RHS (`cho_factor`/`cho_solve`).
          Shared `O(m³)` Cholesky dominates, so this is a modest win — do only when touching this file.
          Scope: the JAX adapter only (not `src/simsopt/field/force.py`).

21. **A1 — Boozer residual dense-Jacobian helper (test/reference surface)** · `src/simsopt_jax/geo/boozer_residual.py:967,977` `boozer_residual_jacobian_composed`
    - [ ] Low priority / API-risk: the `jnp.eye(n_res)` (VJP) and `jnp.eye(n_dofs)` (JVP) dense bases are
          materialized only for the public/test/benchmark derivative surface (no `src/` production caller;
          production exact solves route through the operator-backed adjoint). If addressed, gate any change
          behind the existing parity tests and keep the public signature.

22. **A2 — Stage-2 dynamic min-distance Python loops** · `src/simsopt_jax_adapters/objectives/stage2_target.py:506,536` (called at `:983,:992`)
    - [x] **Jit context RESOLVED (2026-06-23, static inspection):** the calls are inside `_reporting_summary`
          (`:939`), wrapped as a **separate** `reporting_summary = jax.jit(_reporting_summary)` (`:1045`) and
          exposed as a distinct `Stage2ReportingFn` (`:160`, `:1331`) — **not** part of the differentiated
          objective/loss graph (the loss path uses the smooth penalty scan `:496`). So impact = one-time
          compile breadth of a reporting-only jitted fn + its per-call latency, **not** per-optimization-step
          and **not** in the gradient.
    - [ ] Lowest priority. Only worth batching the homogeneous curve groups (T2.1 pattern) **if**
          `reporting_summary` is invoked frequently (e.g., every iteration). Otherwise leave as-is.

### Watch item — no action unless triggered

- **A5** · `src/simsopt_jax/solve/dispatch.py:539` `materialize_dense_linearization` builds a dense
  `jax.jacrev` Jacobian + `jacobian.T @ jacobian` Hessian, but is guarded by a byte cap
  (`max_dense_linearization_bytes`, `:546-554`) and defaults off via the public Optimistix contract.
  - [ ] Re-open only if a production route is found enabling this without a cap.

## Validation Plan

Run targeted suites per change (per-file; suite is not xdist-safe):

- [ ] **T1.2** — `python3 -m pytest tests/solve/test_serial_jax.py tests/objectives/test_constrained.py tests/solve/test_constrained.py -q`; plus `JAX_LOG_COMPILES=1` manual check of single inner-solve compile after warm-up.
- [ ] **T1.1** — `python3 -m pytest tests/geo/test_optimizer_jax_item19.py tests/geo/test_optimizer_jax_reference.py -q`; assert host-vs-device dense build agreement ≤ 1e-12 rel-L2.
- [ ] **T3.1** — `python3 -m pytest tests/geo/test_optimizer_jax_reference.py -q` and any curve-geometry suite; explicit `transfer_guard("disallow")` + `jax.jit` slice probe; assert bit-exact gather.
- [ ] **T2.2 / T2.4** — `python3 -m pytest tests/jax/core/test_pm_optimization_jax_item25.py tests/solve/test_pm_optimization.py tests/solve/test_pm_workflow_jax.py -q` (T2.2 expects bit-identical).
- [ ] **T2.6 / T3.6 (gsco)** — `python3 -m pytest tests/solve/test_wireframe_workflow_jax.py tests/solve/test_wireframe_optimization_jax_item31.py -q`.
- [ ] **T3.2** — `python3 -m pytest tests/core/test_reductions.py -q`; assert 0.0 diff vs current tree.
- [ ] **T2.5** — `python3 -m pytest tests/jax/core/test_boozer_interp_device_cache_and_regular_grid_fused.py tests/jax/core/test_regular_grid_interp_item13.py tests/field/test_interpolated_field_jax_item15.py -q`; + 1 Poincaré/DOPRI5 trajectory spot-check.
- [ ] **T2.1 / A2** — `python3 -m pytest tests/geo/test_surface_objectives_jax.py -q`; assert pair-count unchanged and J/∇J equal to ~1e-12.
- [ ] **#11** — `python3 -m pytest tests/solve/test_wireframe_optimization_jax_item31.py -q`.
- [ ] **T2.3** — `python3 -m pytest tests/field/test_wireframefield_jax_item30.py tests/field/test_wireframefield.py -q`.
- [ ] **T3.3** — `python3 -m pytest tests/mhd/test_vmec_fieldlines_jax.py -q` (re-validate at 1e-13).
- [ ] **T3.4** — `python3 -m pytest tests/mhd/test_bootstrap_jax.py -q` (re-validate JVP-vs-FD oracle; tolerance-matched, not byte-pinned).
- [ ] **T3.5** — `python3 -m pytest tests/mhd/test_vmec_compute_geometry_jax.py tests/mhd/test_vmec_diagnostics.py -q` (VMEC geometry field parity/consistency).
- [ ] **T3.6(force) / A3 / A4 / #3** — `python3 -m pytest tests/field/test_selffieldforces.py -q`.
- [ ] **A1** — `python3 -m pytest tests/geo/test_boozer_residual_jax.py tests/geo/test_boozer_derivatives_jax.py -q`.
- [ ] **Cross-cutting regression** — `python3 -m pytest tests/integration/test_single_stage_objective_parity.py -q` after Phase 1 and Phase 3 (cross-backend J/∇J parity; objective value is env/config-hash sensitive, so compare J/∇J not raw bytes).

## Risks and Mitigations

- Risk: **T3.1** static `slice_in_dim` trips `transfer_guard("disallow")` under jit (the reason the selector
  exists). Mitigation: add a durable guard regression first; fall back to a tracer-safe gather, never the dense matmul.
- Risk: **T2.4 / T3.6 / A4** disturb C++ bit-parity oracles in pm/wireframe/self-force tests.
  Mitigation: opt-in flags only; keep the default path identical; do not algebraically rewrite the proxy.
- Risk: **T2.5** long chaotic DOPRI5 trajectories shadow-diverge from the ~1e-12 einsum reassociation.
  Mitigation: keep strict path behind a parity-debug flag; spot-check a Poincaré run before defaulting fast.
- Risk: **T2.1** mis-grouped batches change the constraint pair set, silently altering the objective.
  Mitigation: assert exact pair count + J/∇J parity at 1e-12 against the pre-change tree.
- Risk: **T1.1** host-vs-device reduction-order delta exceeds tolerance on some shapes.
  Mitigation: bounded agreement assertion (≤1e-12 rel-L2); the device lane already accepts ~1e-16.
- Risk: **T3.3 / T3.4 / T3.5** alter MHD numerical routes and drift from CPU/JAX oracles.
  Mitigation: keep current dense/pre-broadcast forms available as test oracles until the replacement passes
  the named VMEC/bootstrap suites at their existing tolerances.
- Risk: JAX cache accumulation across the non-xdist suite produces spurious OOM/aborts.
  Mitigation: run per-file; rely on the conftest per-test `jax.clear_caches`.

## Completion Criteria

- [ ] Phase 1 (T1.2, T1.1) merged; single inner-solve compile confirmed; host Hessian build delegates to the
      chunked device sibling; targeted suites green; single-stage J/∇J parity test green.
- [ ] Phase 2 bit-exact items merged with 0.0 (or ≤1e-12 where reduction-order changes) diffs on their suites.
- [ ] Phase 3 (T2.5 widened incl. classifier, T2.1, #11) merged; Stage-2 pair count unchanged; Cartesian/cyl
      + classifier tracing default to the fast contraction with strict reachable behind a flag.
- [ ] Phase 4 items either merged with their VMEC/bootstrap/self-force parity gates preserved, merged behind
      opt-in flags where the current default must stay identical, or explicitly deferred with a one-line note here.
- [ ] A5 watch-item documented as deferred (no production uncapped route found) or escalated.
- [ ] This plan updated: each task checkbox reflects merged/deferred state; no answered Open Question left open.

## Resolved Questions

- **A2 — RESOLVED 2026-06-23 (static inspection):** the min-distance loops are inside `_reporting_summary`
  (`:939`), a **separate** `jax.jit(_reporting_summary)` (`:1045`) exposed as `Stage2ReportingFn` — not in the
  differentiated objective graph. Impact = one-time compile breadth of a reporting-only fn; batch only if
  reporting is invoked per-iteration. Lowest priority.

## Open Questions / Execution Gates

- **T3.1 transfer-guard gate:** static inspection cannot prove that the planned `slice_in_dim` replacement is
  safe under the repo's active JAX transfer-guard settings. The prior scratch probe paths named in an earlier
  draft are not present in this checkout, so implement a durable regression before deleting the selector helpers.
- **Local runtime gate:** this doc-review pass did not rerun local JAX imports or pytest. The validation plan
  lists the commands to run after implementation; do not mark a task complete from this document alone.
