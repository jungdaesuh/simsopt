# Bucket 2 Audit — M3 composed derivatives + M4 JAX Boozer solver

Date: 2026-04-25
Auditor: max-effort test-quality audit (Claude Opus 4.7)
Branch: gpu-purity-stage2-20260405
Total tests in scope: **306** (residual=14, derivatives=21, surface=196, surface_private=60, label=6, converters=9)

Reference contract: `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` is the SSOT.
- `direct_kernel`: `rtol=1e-10, atol=1e-12` (same-state, vector parity required)
- `ls_wrapper_gradient`: `rtol=1e-10, atol=1e-12` (same-state)
- `derivative_heavy`: 1st-deriv `rtol=1e-8, atol=1e-10`; 2nd-deriv `rtol=1e-6, atol=1e-8`
- `exact_well_conditioned_adjoint`: `adjoint_rtol=1e-6, adjoint_atol=1e-8, residual_rel_tol=1e-10`, vector parity required
- `exact_ill_conditioned_adjoint`: residual-only `1e-10`, **vector parity NOT required**
- `branch_stable_resolve`: core_value `rtol=1e-6, atol=1e-7`; derived_value `rtol=5e-5`
- `fd_gradient`: directional FD `rtol=1e-5, atol=1e-7`

---

## 1. Per-file summary table

| File | Total | Tautological | Loose | Weak | Meaningless | Mocked | VJP-drift | Exact-lane viol. | Well-tightened | Priority |
|------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|--------|
| tests/geo/test_boozer_residual_jax.py | 14 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 11 | LOW |
| tests/geo/test_boozer_derivatives_jax.py | 21 | 4 | 2 | 0 | 1 | 0 | 0 | 0 | 14 | MED |
| tests/geo/test_boozersurface_jax.py | 196 | 12 | 7 | 35 | 18 | 14 | 5 | 2 | ~103 | **HIGH** |
| tests/geo/test_boozersurface_jax_private.py | 60 | 1 | 1 | 22 | 4 | 0 | 0 | 0 | 32 | MED |
| tests/geo/test_label_constraints_jax.py | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | LOW |
| tests/geo/test_optimizer_result_converters.py | 9 | 0 | 0 | 9 | 9 | 9 | 0 | 0 | 0 | LOW (smoke) |
| tests/geo/boozersurface_jax_test_helpers.py | n/a (helper) | — | — | — | — | — | — | — | — | **HIGH (root cause for M4 mockedness)** |

Key cluster verdict: M3 derivative tests (`test_boozer_derivatives_jax.py`) are mostly well-tightened JVP/FD contracts on a real torus + analytic coil set. M4 surface tests (`test_boozersurface_jax.py`) split into ~30% real-physics tests and ~70% routing/contract tests of the LS/exact orchestration; many of those are weak-assertion, mocked, or tautological. The `boozersurface_jax_test_helpers.py` module is the source of mocked physics — its `_MockSurface`, `_MockBiotSavart`, `_MockVolumeLabel` are SHALLOW MOCKS and most class-level test results carrying these mocks should be treated as smoke-level.

---

## 2. Top issues (table)

| # | file:line | Test name | Class | Evidence (literal quote) | Tightening |
|---|-----------|-----------|-------|--------------------------|------------|
| 1 | test_boozersurface_jax.py:5945 (`test_finite_unsuccessful_state_with_adjoint_contract_is_rejected`) is OK; **issue is** test_boozersurface_jax.py:5946+5962+5988 cluster — `_make_mock_boozer_surface` uses `_MockVolumeLabel.J() -> 0.0` and `_MockBiotSavart` carries no real field state | `test_ensure_solved_logs_*` | OVERLY MOCKED + WEAK | `_MockVolumeLabel` returns 0.0 unconditionally; the "label J() = 0" prevents any volume-constraint physics check | Replace `_MockVolumeLabel` with a real `Volume(surface)` (label_constraints_jax.volume_jax) and assert `J()` matches `2π²Rr²`; or mark these explicitly as "log-format smoke". |
| 2 | test_boozersurface_jax.py:802-822 `test_penalty_gradient_fd` | TestComposedPenaltyObjective | LOOSE / WEAK COVERAGE | `for idx in [0, len(d["x"]) // 2, -2, -1]: ... rtol=1e-6, atol=1e-10` — only 4 components checked, single ε=1e-6 (no Taylor ladder) | Use `_assert_composed_penalty_gradient_contract` from test_boozer_derivatives_jax.py (full ε-ladder + JAX `check_grads`) at `derivative-heavy` lane (rtol=1e-8). |
| 3 | test_boozersurface_jax.py:828-841 `test_bfgs_rosenbrock` | TestOptimizerAdapter | MEANINGLESS PHYSICS | `np.testing.assert_allclose(result.x, jnp.array([1.0, 1.0]), atol=_ROSENBROCK_SOLUTION_ATOL)` — Rosenbrock is unrelated to Boozer physics | Either move to a generic optimizer test file or replace with a Boozer LS quadratic surrogate that has known minimum. |
| 4 | test_boozersurface_jax.py:970-982 `test_newton_polish_quadratic` | TestOptimizerAdapter | MEANINGLESS PHYSICS | `A = jnp.array([[2.0, 0.5], [0.5, 3.0]])` — generic 2x2 system, unrelated to Boozer Newton | Same as #3 — generic optimizer test. Move to test_optimizer_jax.py. |
| 5 | test_boozersurface_jax.py:1124-1137 `test_newton_exact_linear_system` | TestOptimizerAdapter | MEANINGLESS PHYSICS + WEAK | `A = jnp.array([[3.0, 1.0], [1.0, 4.0]])` — purely synthetic; only checks `np.testing.assert_allclose(result["x"], x_exact, atol=1e-12)` for a linear system the solver finishes in 1 step | Move to optimizer test file; for Boozer-relevant exact path, add a real composed-residual test that checks `‖r‖ ≤ 1e-12` after Newton on a small-but-real torus. |
| 6 | test_boozersurface_jax.py:1253-1269 `test_newton_polish_reduces_gradient` | TestNewtonPolishBoozer | WEAK ASSERTION | `assert newton_grad_norm <= bfgs_grad_norm + 1e-15` — only direction-of-improvement, no magnitude target; uses mock physics | Tighten to assert post-Newton `‖grad‖ < 1e-10` (Boozer LS post-polish should be near machine eps) on a real (non-mock) torus. |
| 7 | test_boozersurface_jax.py:1290-1299 `test_penalty_with_toroidal_flux` | TestToroidalFluxLabel | WEAK ASSERTION | `val = case["objective"](case["x"]); assert val.shape == (); assert float(val) >= 0.0` then `grad.shape == case["x"].shape` — only shape and non-negativity | Add gradient FD for toroidal-flux path with `derivative-heavy` tolerance; the toroidal-flux label is the highest-stakes path because it requires `biot_savart_A` plus `gammadash2`. |
| 8 | test_boozersurface_jax.py:1305-1312 `test_lbfgs_reduces_objective` | TestLBFGSMethod | WEAK | `assert float(result.fun) < val_init` — direction only, no magnitude | Assert `‖grad‖ < 1e-8` post-LBFGS on real surface, OR scope this as smoke. |
| 9 | test_boozersurface_jax.py:1339-1348 `test_penalty_with_area_label` | TestAreaLabelPath | WEAK | Same shape+non-negativity pattern as #7 | Add area-label FD/Taylor coverage; area is currently UNTESTED at the gradient level inside the composed objective. |
| 10 | test_boozersurface_jax.py:2594-2604 `test_run_code_ls_converges` | TestBoozerSurfaceJAXClass | WEAK + MOCKED | `assert "PLU" in res; assert "vjp" in res` — only checks dict-key presence; uses `_MockBiotSavart` with circular coils | Add `assert res["fun"] < 1e-8` (penalty objective at solution) and `res["jacobian"]`-norm bound. |
| 11 | test_boozersurface_jax.py:3390-3394 `test_run_code_idempotent` | TestBoozerSurfaceJAXClass | WEAK / TAUTOLOGICAL | `assert booz.run_code(iota=0.3, G=0.05) is None` — only tests dirty-flag plumbing | Acceptable as smoke; rename to `test_run_code_dirty_flag_idempotent`. |
| 12 | test_boozersurface_jax.py:3396-3410 `test_run_code_sdofs_matches_implicit_path` | TestBoozerSurfaceJAXClass | TAUTOLOGICAL | Compares `res_sdofs["iota"]` to `res_ref["iota"]` from same `_make_mock_boozer_surface()` and same solver — both branches end up calling identical code | Either prove the two branches enter different control flow (assert different intermediate state) or scope as smoke. |
| 13 | test_boozersurface_jax.py:3946-3976 `test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity` | TestBoozerSurfaceJAXClass | WELL-TIGHTENED on adjoint vector parity but **fixture is fully synthetic**: lines 139-168 set `A = diag + small triangular`, `booz.res["PLU"]` is fabricated. The `_make_exact_residual` is `lambda x: A @ x`, NOT the real Boozer exact residual. | OVERLY-MOCKED for "exact-well-conditioned-adjoint" lane — passes with fixture data only | Layer in a real (small) torus exact solve with verified well-conditioning before the dense-comparison block. The synthetic A is fine for adjoint-state plumbing but does not exercise actual Boozer Jacobian conditioning. |
| 14 | test_boozersurface_jax.py:5407-5435 (3 tests `test_exact_mask_*`) | TestBoozerSurfaceJAXExactPath | WEAK | `assert mask.dtype == np.bool_`, `assert mask.shape == (3 * nphi * ntheta,)`, `masked_r = res["residual"][res["mask"]]; assert masked_r.ndim == 1` — pure shape/dtype | Acceptable as schema tests but should be one combined test; the mask CONTENT (which entries are masked under stellsym) is the load-bearing invariant and is only checked by the CPU-vs-JAX `TestStellsymMaskCPUJAXParity` block (line 6481+). |
| 15 | test_boozersurface_jax.py:5443-5469 `test_exact_*_aborts_adjoint_state` (2 tests) | TestBoozerSurfaceJAXExactPath | WEAK | Only checks `res["PLU"] is None`, `res["vjp"] is None`, `res["mask"] is None` and dofs unchanged | Add residual-norm assertion: failed-newton must report `‖res["residual"]‖` consistent with the failure (NaN or large finite). Currently a successful path that erroneously cleared adjoint metadata would silently pass. |
| 16 | test_boozersurface_jax.py:5453 — `_run_mock_exact_boozer_success` patches `_bsj.newton_exact` to return a synthetic `_successful_exact_newton_result` (jacobian = identity) | TestBoozerSurfaceJAXExactPath cluster (~14 tests) | OVERLY MOCKED | `_successful_exact_newton_result` returns `"jacobian": jnp.eye(n, dtype=x0.dtype)` — Jacobian is identity, residual is zero, with NO connection to the actual Boozer residual | These tests validate result-dict plumbing not physics. Add at least ONE end-to-end exact-solve test on a real surface where `newton_exact` is NOT patched (similar to the FD Taylor test at 6105-6128 but going through `run_code(...)`). |
| 17 | test_boozersurface_jax.py:6018-6031 `TestMixedQuadratureBoozer` | MIXED | WEAK | `test_instantiation`: `assert len(booz.coil_groups) == 2`. `test_run_code_ls_converges`: `assert res["success"]` — no value check. `test_penalty_matches_uniform`: `np.testing.assert_allclose(res_mixed["fun"], res_uniform["fun"], rtol=2e-3, atol=2e-6)` | LOOSE — 2e-3 is 5 orders looser than `derivative-heavy` 1e-8. The 2e-3 reflects quadrature truncation but the test never compares to a reference penalty value. | Add a fixed-state `BiotSavartJAX(B)` evaluation parity test between mixed and uniform at matched quadrature points (should agree to `direct-kernel rtol=1e-10`) before relaxing for full-solve. |
| 18 | test_boozer_residual_jax.py:300-315 `test_grad_iota` | TestBoozerResidualGradient | LOOSE | `eps = 1e-6; ... rtol=1e-5` — central FD with single ε is `O(eps²)` ~ 1e-12, but tolerance is 1e-5 | This is fine for sanity but is at the `fd-gradient` lane (1e-5). For `derivative-heavy`, tighten to `rtol=1e-8` with ε-ladder (the M3 test `_assert_scalar_grad_matches` at test_boozer_derivatives_jax.py:224-248 already uses rtol=1e-10). |
| 19 | test_boozer_residual_jax.py:412-428 `test_hessian_matches_grad_fd` | TestBoozerResidualHessian | LOOSE | `eps = 1e-5; ... rtol=1e-4` — only checks 2 of 4 Hessian entries; rtol=1e-4 vs `derivative-heavy` second-derivative `rtol=1e-6` | Tighten to `rtol=1e-6` and check ALL 4 entries (or check off-diagonals symmetric within rtol). |
| 20 | test_boozer_residual_jax.py:449-471 `test_surface_dof_gradient_is_zero` | TestBoozerResidualM1Limitations | TAUTOLOGICAL by design | `np.testing.assert_allclose(host_array(grad[:nsurfdofs]), 0.0, atol=1e-30)` — checks that M1 returns zero, which is the documented behavior | OK — explicitly documented as scope-limit, NOT a bug. Keep but rename to `test_m1_returns_zero_for_surface_dofs_documents_scope_limit`. |
| 21 | test_boozer_derivatives_jax.py:447-458 `test_dgamma_shape` (and stellsym `test_dgamma_stellsym_fewer_dofs` 544-557) | TestDgammaByDcoeff | WEAK | `assert J.shape == (self.nphi, self.ntheta, 3, ndofs)` and `assert ndofs_stellsym < ndofs_full` | Fine as schema tests. The actual contract is in `test_dgamma_fd` etc. |
| 22 | test_boozer_derivatives_jax.py:597-603 `test_gradient_fd` (TestBoozerPenaltyGradComposed) | WELL-TIGHTENED | `_assert_composed_penalty_gradient_contract` calls JAX `check_grads(... atol=1e-6, rtol=1e-5, eps=1e-6)` AND `_assert_scalar_grad_matches(rtol=1e-10, atol=1e-12)` | The grad-matches part is at `direct-kernel` precision (1e-10). Solid. |
| 23 | test_boozer_derivatives_jax.py:712-715 `test_hessian_symmetry` | TestBoozerHessianComposed | WEAK | `np.testing.assert_allclose(np.array(H), np.array(H.T), atol=1e-12)` — symmetry does not check correctness, only that JAX `hessian` produces a symmetric matrix (which follows from `jax.hessian` definition for scalar functions, modulo floating-point) | Acceptable as a fast smoke; the load-bearing test is the FD test at 717-731. The `jax.hessian` of a scalar is symmetric by definition, so this is a near-tautology. |
| 24 | test_boozer_derivatives_jax.py:733-748 `test_hessian_taylor_convergence` | WELL-TIGHTENED | `best_defect_tol=1e-8, min_observed_order=2.0` — confirms quadratic Taylor convergence | Solid. |
| 25 | test_boozer_derivatives_jax.py:917-927 `test_gradient_weighted_fd` | TestComposedWeightInvModB | LOOSE | Comment: `# Near-pole 1/|B| weighting is the remaining FD-sensitive path here.` `check_grads_atol=5e-6, check_grads_rtol=5e-5, check_grads_eps=3e-6` — relaxes from baseline 1e-6/1e-5/1e-6 | The relaxation IS justified (1/|B| amplifies near-pole noise) but the comment shouldn't cover it; explicitly map to `fd-gradient` lane (rtol=1e-5) and document why. |
| 26 | test_boozer_derivatives_jax.py:842-844 `test_coil_vjp_currents_fd` and 867-878 `test_coil_vjp_geometry_scalarization` | TestBoozerResidualCoilVJP | WELL-TIGHTENED on `(lm, gamma=, xphi=, xtheta=, coil_arrays=, iota=, G=, weight_inv_modB=)` signature, scalar grad rtol=1e-10, atol=1e-12 | Solid; matches the M3 explicit signature from `boozer_residual_coil_vjp`. NOT the M4 `(lm, booz_surf, iota, G)` signature. |
| 27 | test_boozersurface_jax.py:5536-5554 `test_ls_vjp_returns_correct_shapes` | TestVJPHooks | WEAK + VJP CONVENTION DRIFT (mild) | `d_coil_arrays, coil_indices = vjp_fn(jnp.asarray(lm), booz, iota_sol, G_sol)` — calls JAX VJP signature `(lm, booz_surf, iota, G)`. Then `assert d_g.shape == g.shape` etc. — only shape | Add value parity: compare against `_boozer_ls_coil_vjp(lm, booz_surf, iota, G)` from the source module (the same function). Currently this test only validates plumbing and could pass with a transposed wrong-axis cotangent. |
| 28 | test_boozersurface_jax.py:5682-5715 `test_ls_group_vjp_toroidal_flux_matches_full_vjp` | TestVJPHooks | WELL-TIGHTENED but compares JAX-vs-JAX (streaming vs non-streaming) | `np.testing.assert_allclose(streamed_arr, full_arr, rtol=1e-10, atol=1e-10)` — both come from JAX module | OK but not an oracle test. The streaming-vs-bulk comparison validates an internal refactor invariant; physics validity rests on `test_ls_reduced_directional_requires_spatial_field_derivatives` (5600-5655). |
| 29 | test_boozersurface_jax.py:5600-5655 `test_ls_reduced_directional_requires_spatial_field_derivatives` | TestVJPHooks | WELL-TIGHTENED + LOAD-BEARING | `np.testing.assert_allclose(full_directional, fd_directional, rtol=1e-8, atol=1e-10)` and `assert abs(float(dropped_directional - fd_directional)) > 1e-7` | Excellent — this is a true regression test that proves the dB/dX term must be present. `derivative-heavy` lane. |
| 30 | test_boozersurface_jax.py:5731-5759 `test_add_G_current_cotangent_matches_abs_vjp_at_zero_current` | TestVJPHooks | WELL-TIGHTENED edge case | `rtol=1e-14, atol=1e-14` against `jax.vjp(compute_G_from_currents, currents)` | Strong — pins the abs-subgradient convention at 0. Direct-kernel precision. |
| 31 | test_boozersurface_jax.py:6105-6127 `TestBoozerExactConstraintsJacobianTaylor` (2 tests) | WELL-TIGHTENED | `ratio_bound=0.55` Taylor convergence over `2^-7..2^-19` | Strong. Real exact-residual Jacobian (not patched). |
| 32 | test_boozersurface_jax.py:6202-6208 `test_gradient_taylor` (parametrized) and 6216-6230 `test_bfgs_reduces_objective` | TestParametrized* | MIXED | Gradient: ratio=0.55 ε-ladder Taylor (well-tightened). BFGS: `assert val_final < val_init` (weak). | Tighten BFGS to assert `val_final < 1e-3 * val_init` or absolute tolerance; current pass criterion is "any reduction at all". |
| 33 | test_boozersurface_jax.py:6371-6389 `test_penalty_gradient_taylor_matrix` | TestUpstreamFactoryBoozerMatrix | WELL-TIGHTENED | Real NCSX surfaces, ratio=0.55 ε-ladder | Solid. |
| 34 | test_boozersurface_jax.py:6393-6405 `test_penalty_value_and_gradient_cpu_parity_tensor_matrix` | TestUpstreamFactoryBoozerMatrix | WELL-TIGHTENED | Direct CPU vs JAX parity: `_UPSTREAM_PENALTY_VALUE_PARITY_RTOL=1e-5`, gradient max-abs `<1e-2`, rel-norm `<2e-3`. | LOOSE for `direct-kernel` (would expect 1e-10). The 1e-5 reflects integration-of-residual sensitivity in NCSX; OK because it's a `derivative-heavy` 1st-derivative comparison via FD-equivalent. Could be tighter on value (1e-10 if same quadrature). |
| 35 | test_boozersurface_jax.py:6440-6448 `test_run_code_exact_accepts_deferred_xyztensor_contract` | TestUpstreamFactoryBoozerMatrix | WEAK | `assert res["success"] is True; assert res["type"] == "exact"` with patched `_patched_exact_newton_result(success=True)` | Pure plumbing test — does not validate that the deferred tensor surface produces a correct residual. |
| 36 | test_boozersurface_jax_private.py:1232-1244 `test_bfgs_ondevice_respects_zero_iteration_budget` | TestOptimizerAdapterPrivate | WEAK | `assert result.nit == 0; assert result.status == 1; assert result.success is False` | OK as smoke; not pretending to be physics. |
| 37 | test_boozersurface_jax_private.py:1248-1260 `test_bfgs_ondevice_zero_gradient_converges_immediately` | TestOptimizerAdapterPrivate | TAUTOLOGICAL borderline | Initialize at zero gradient, then verify the solver converges immediately. | Smoke; OK. |
| 38 | test_boozersurface_jax_private.py:1335-1350 `test_bfgs_ondevice_is_deterministic` | TestOptimizerAdapterPrivate | TAUTOLOGICAL | `np.testing.assert_allclose(np.asarray(first.x), np.asarray(second.x))` — runs the same JAX-jit twice; trivially equal | OK as a determinism smoke. Could mark as `@pytest.mark.smoke`. |
| 39 | test_optimizer_result_converters.py:116-305 (all 9 tests) | All | OVERLY MOCKED | All tests build `SimpleNamespace`s and call `converters._private_lbfgs_result_to_optimize_result(...)` — pure host-boundary type conversion tests with no optimizer behavior | Correct as host-boundary unit tests; do NOT reclassify as physics tests. Leave as-is. |
| 40 | boozersurface_jax_test_helpers.py:521-612 `_MockSurface`, `_MockBiotSavart`, `_MockVolumeLabel` | helper | OVERLY-MOCKED at root | `_MockVolumeLabel.J() -> 0.0` (constant); `_MockSurface.get_stellsym_mask()` returns all-True regardless of stellsym; `_MockBiotSavart` only stores coils, never re-evaluates field | These mocks are used by ~70% of `TestBoozerSurfaceJAXClass` tests. The fact that `_MockVolumeLabel.J()` is identically 0 means the *constraint* term in the penalty is `targetlabel - 0 = constant`, so the gradient of the constraint w.r.t. surface DOFs is zero — half of the penalty physics is silenced. |

---

## 3. Missing coverage

Mathematical / physical invariants that should be tested but aren't (or are tested only on synthetic / mocked data):

- **Exact-vs-LS path equivalence at the same solved state.** When the LS solver converges to a true minimum where the gradient equals zero, the resulting `(sdofs, iota, G)` should also satisfy the exact-residual system to within the ill-conditioning of that residual. No test verifies this. Add: solve LS on real torus to high tolerance, evaluate exact residual, assert `‖r‖ < 1e-6` (relative to magnitude of `G·B`).

- **Composed Boozer residual scaling property.** `boozer_residual_vector(α·G, iota, α·B, xphi, xtheta, weight_inv_modB=False)` should scale as `α²` (since each term is `G·B - |B|² · tang`). No homogeneity check exists. Add at `direct-kernel` precision (rtol=1e-12).

- **Iota gauge sensitivity.** `boozer_residual` should be invariant under iota → iota + 2π/q for rational q on closed orbits in the periodic angle convention. Not tested.

- **Stellsym-on stellsym-off composed gradient parity at a stellarator-symmetric configuration.** A stellsym surface with stellsym=True and the same DOF unrolled to non-stellsym should produce identical penalty values and gradients projected to stellsym subspace. No test.

- **Volume-vs-area label cross-validation.** Current tests use volume label OR area label OR toroidal-flux label, but no test sets two label types and confirms the JAX path matches expected sign+magnitude relations between them on a torus.

- **Ill-conditioned exact path: residual+failure-only contract.** No test in `TestBoozerSurfaceJAXExactPath` constructs an actually ill-conditioned exact problem and verifies (a) the operator solve fails with a `failure_category` flag, (b) the residual norm is reported correctly, and (c) NO vector parity assertion is made. Currently the well-conditioned synthetic fixture (line 131-196) is the only exact-adjoint coverage and it is well-conditioned by construction.

- **Dense-vs-operator adjoint disagreement detection.** `test_exact_adjoint_dense_metadata_does_not_change_operator_runtime` (line 3978) confirms operator runs but does not assert that operator and dense PLU adjoints would disagree if dense factors went stale. Add a positive test where the dense PLU is intentionally stale (different x*) and the operator-backed solve still wins.

- **VJP signature regression guard.** The M3 (`boozer_residual_coil_vjp(adjoint, gamma=..., xphi=..., xtheta=..., coil_arrays=..., iota=..., G=...)`) and M4 (`_boozer_ls_coil_vjp(lm, booz_surf, iota, G)`, `_boozer_exact_coil_vjp(lm, booz_surf, iota, G)`) signatures differ. Tests at `test_boozersurface_jax.py:5546-5554` use the M4 4-arg form. There is no explicit guard that catches a drift back to the CPU `(lm, booz_surf)` signature — `test_run_code_rejects_bad_group_vjp_signature` (5761-5776) only checks vjp_groups arity, not the two distinct `vjp` signatures. Add a `inspect.signature` regression for both `_boozer_exact_coil_vjp` and `_boozer_ls_coil_vjp`.

- **Newton-polish-after-LS gradient norm absolute target.** `test_newton_polish_reduces_gradient` (1253-1269) only checks "Newton ≤ BFGS". The actual Boozer LS post-polish should reach `‖grad‖ < 1e-10` on a real (non-mock) surface. No such test exists.

- **`boozer_residual_jacobian_composed` dimension consistency.** Tests check shape `(n_res, n_dofs)`, but no test verifies that for `optimize_G=True`, column n_dofs−1 is the partial w.r.t. G (which equals `B[..., :].ravel()`). A simple closed-form for the iota and G columns is missing.

- **Hessian curvature-of-objective at a true minimum.** No test verifies that at a converged Boozer LS minimum, the eigenvalues of `jax.hessian(boozer_penalty_composed)` are non-negative (PSD up to numerics). This is the natural sanity check for the Newton polish.

- **Toroidal-flux label gradient FD inside `_boozer_penalty_objective`.** `test_label_constraints_jax.py` covers `toroidal_flux_jax` standalone, and `test_boozer_derivatives_jax.py` covers volume-only composed. No `derivative-heavy` lane FD test of toroidal-flux composed penalty gradient.

- **`_compute_G_from_currents` sign convention vs CPU.** `compute_G_from_currents` uses `μ₀ Σ|I_k|`. No CPU/JAX parity test against the upstream `BoozerSurface._initial_G_from_currents` exists in this bucket (only end-to-end through `_upstream_initial_G` helper at boozersurface_jax_test_helpers.py:107-109, which is itself a JAX-side reproduction).

- **`reduction_mode="strict_oracle"` for composed objectives.** The strict_oracle mode is tested only on synthetic dynamic-range data in `test_strict_oracle_scalar_mode_matches_high_precision_reference`. No test verifies that the composed penalty objective accepts `reduction_mode="strict_oracle"` and produces a more precise scalar than `default` on a near-floor configuration.

- **`_DEFAULT_MAX_DENSE_JACOBIAN_BYTES` rationale gate.** `test_ls_surface_exact_newton_has_default_dense_jacobian_ceiling` (4468-4497) checks that the option is forwarded but not what value triggers "scaling_limit". A boundary test (n that produces exactly the ceiling) is missing.

---

## 4. Tightening playbook (P0 — top 12)

These are the highest-impact, lowest-effort changes. Listed in priority order.

1. **(test_boozersurface_jax.py:802-822, `test_penalty_gradient_fd`)** Replace 4-component manual ε=1e-6 FD with the `_assert_composed_penalty_gradient_contract` helper from `test_boozer_derivatives_jax.py`. ~10 lines.

2. **(boozersurface_jax_test_helpers.py:595-599, `_MockVolumeLabel`)** Make `_MockVolumeLabel.J()` actually call `surface_volume(gamma, normal)` so the constraint term has nonzero gradient. Half the M4 penalty tests are silently null-testing the constraint contribution. ~15 lines.

3. **(test_boozersurface_jax.py:1290-1299, `test_penalty_with_toroidal_flux`)** Add a Taylor-test gradient FD for the toroidal-flux composed penalty. The toroidal-flux path is the most code-distinct (uses `biot_savart_A` not `_B`). Use `_assert_composed_penalty_gradient_contract` style. ~20 lines.

4. **(test_boozersurface_jax.py:1339-1348, `test_penalty_with_area_label`)** Same as #3 for area label.

5. **(test_boozersurface_jax.py:828-841 + 970-982 + 1124-1137 + 1230-1247)** Move the four pure-numerics optimizer tests (Rosenbrock, generic newton_polish_quadratic, newton_exact_linear_system, newton_exact_traceable_operator_only_path_remains_jittable) to `test_optimizer_jax_generic.py`. They aren't Boozer tests. Frees ~80 lines in the M4 file.

6. **(test_boozer_residual_jax.py:300-331, `test_grad_iota`/`test_grad_G`)** Tighten from `eps=1e-6, rtol=1e-5` to ε-ladder + `rtol=1e-8` matching `derivative-heavy`. Use `check_grads`. ~15 lines.

7. **(test_boozer_residual_jax.py:412-428, `test_hessian_matches_grad_fd`)** Tighten from `rtol=1e-4` to `rtol=1e-6`; check all 4 entries (currently only [0,0] and [0,1]).

8. **(test_boozersurface_jax.py:5536-5554, `test_ls_vjp_returns_correct_shapes`)** Add value parity assertion — call `_boozer_ls_coil_vjp` directly (the M4 signature) and compare to `res["vjp"]` output at `direct-kernel` rtol=1e-12. The current shape-only test would pass with a transposed wrong-axis cotangent.

9. **(test_boozersurface_jax.py: TestBoozerSurfaceJAXExactPath cluster)** Add ONE end-to-end exact-Newton test on a real torus (no `_patched_exact_newton_result`) that asserts (a) `‖res["residual"]‖ < 1e-10`, (b) `‖J^T r‖ < 1e-10`, (c) `res["jacobian_materialized"] is True`. Exposes any silent regression in `_select_exact_residual_fn` or `_make_exact_residual`.

10. **(MISSING) Add an ill-conditioned exact-path test** — construct a low-iota or near-axis configuration (Boozer Jacobian has known ill-conditioning), assert `failure_category="scaling_limit"` OR operator residual `≤1e-10`, AND make NO vector-parity claim. This is required to actually exercise the `exact-ill-conditioned-adjoint` lane contract; the lane is currently unexercised.

11. **(MISSING) Add `inspect.signature` regression for VJP entrypoints** — pin both `_boozer_exact_coil_vjp` and `_boozer_ls_coil_vjp` to `(lm, booz_surf, iota, G[, weight_inv_modB])`. Ten-line test catches the historical CPU-vs-JAX signature drift.

12. **(test_boozersurface_jax.py:6018-6031, `TestMixedQuadratureBoozer`)** Replace `assert res["success"]` with a fixed-state field-value parity assertion at `direct-kernel rtol=1e-10` between mixed-quad coil set and uniform-quad coil set evaluated at matching points. Convergence is good but means little if the mixed-vs-uniform B field disagrees at the residual floor.

---

## 5. VJP convention audit

Each test that touches a `vjp` callable, classified by the calling signature:

### M3 layer (`boozer_residual_coil_vjp` — explicit args)
Signature: `(adjoint, *, gamma, xphi, xtheta, coil_arrays, iota, G, weight_inv_modB=False)`

| Test | File:line | Signature observed | OK? |
|------|-----------|---------------------|-----|
| `test_coil_vjp_currents_fd` | test_boozer_derivatives_jax.py:842 | `boozer_residual_coil_vjp(adjoint, gamma=..., xphi=..., xtheta=..., coil_arrays=..., iota=..., G=..., weight_inv_modB=False)` (line 808-817) | OK — exact M3 signature |
| `test_coil_vjp_shapes` | test_boozer_derivatives_jax.py:846 | Same M3 keyword form | OK |
| `test_coil_vjp_geometry_scalarization` | test_boozer_derivatives_jax.py:872 | Same M3 keyword form | OK |

### M4 LS path (`_boozer_ls_coil_vjp` and `res["vjp"]` in LS results)
Signature: `(lm, booz_surf, iota, G, weight_inv_modB=True)`

| Test | File:line | Signature observed | OK? |
|------|-----------|---------------------|-----|
| `test_ls_vjp_returns_correct_shapes` | test_boozersurface_jax.py:5548 | `vjp_fn(jnp.asarray(lm), booz, iota_sol, G_sol)` — 4-arg M4 | OK |
| `test_ls_group_vjp_uses_grouped_spec_path` | test_boozersurface_jax.py:5581 | `vjp_groups_fn(lm, booz, res["iota"], res["G"])` — 4-arg | OK |
| `test_ls_group_vjp_does_not_route_through_full_grouped_vjp` | 5596 | Same | OK |
| `test_ls_group_vjp_toroidal_flux_matches_full_vjp` | 5692-5705 | `_bsj._boozer_ls_coil_vjp(lm, booz, iota, G)` — 4-arg M4 | OK |
| `test_ls_group_vjp_repeated_call_reuses_field_kernel_cache` | 5725, 5727 | 4-arg | OK |
| `test_ls_group_vjp_detects_stale_reuse_after_resolve` | 5787 | 4-arg | OK |

### M4 exact path (`_boozer_exact_coil_vjp` and `res["vjp"]` in exact results)
Signature: `(lm, booz_surf, iota, G)`

| Test | File:line | Signature observed | OK? |
|------|-----------|---------------------|-----|
| `test_exact_result_dict_keys` | test_boozersurface_jax.py:4446-4447 | Stores `res["vjp"] is _boozer_exact_coil_vjp` (function-identity check); does NOT call vjp | OK by inspection — does not exercise |
| `test_exact_invalid_newton_iterate_aborts_adjoint_state` | 5443 | Asserts `res["vjp"] is None` after failure | OK |
| `test_exact_unsuccessful_finite_newton_exit_aborts_adjoint_state` | 5457 | Same | OK |

### Adjoint-state callbacks (operator-only contract)

| Test | File:line | Calls | Lane match |
|------|-----------|-------|------------|
| `test_exact_well_conditioned_operator_adjoint_matches_dense_reference_and_plu` | 3930 | `solve_transpose_with_status(rhs)` then asserts `operator_adj` ≈ `jax_dense_adj` ≈ `plu_adj` at `exact-well-conditioned-adjoint` rtol=1e-6, atol=1e-8, residual≤1e-10 | OK — explicit `parity_ladder_tolerances("exact-well-conditioned-adjoint")` |
| `test_exact_well_conditioned_operator_adjoint_cpu_gpu_same_state_parity` | 3946 | Same lane | OK on lane; **fixture is synthetic A=diag**, not real Jacobian |
| `test_exact_adjoint_dense_metadata_does_not_change_operator_runtime` | 3978 | Asserts ONE operator solve happens, dense factors are metadata-only | OK |
| `test_get_adjoint_runtime_state_exact_jacobian_uses_host_tolerance_boundary` | 3831 | Asserts solved∼rhs at `exact-well-conditioned-adjoint` rtol; uses fully patched `_solve_jacobian_system_with_status` returning `rhs` unchanged | TAUTOLOGICAL — when the patched solve returns `rhs`, asserting `solved≈rhs` is `X==X`. The lane numerics tolerance is correct; the test does not validate the operator. |

### Critical findings
- **No CPU `(lm, booz_surf)` signature is in use.** All M4 `vjp` calls match the JAX `(lm, booz_surf, iota, G)` convention.
- **No silent broadcasting/positional accident detected.** The M4 4-arg signature is consistent across the test suite.
- **One TAUTOLOGICAL fixture-vs-fixture adjoint test** (line 3831, `test_get_adjoint_runtime_state_exact_jacobian_uses_host_tolerance_boundary`) — patched solver returns its rhs and the test then asserts solved==rhs. This validates plumbing only.
- **No test asserts the adjoint signature itself.** Adding `assert inspect.signature(_boozer_ls_coil_vjp).parameters.keys() == {"lm","booz_surf","iota","G","weight_inv_modB"}` (and similar for exact) would catch silent drift.

---

## Genuinely solid clusters

Honest credit where due:

- `test_label_constraints_jax.py` — all 6 tests use real Taylor-test Q/2 ratio convergence with proper ε-ladder. Toroidal-flux invariance, gradient FD, Hessian FD, and coil-DOF FD. Strong.
- `test_boozer_derivatives_jax.py::TestDgammaByDcoeff*` and `TestBoozerPenaltyGradComposed::test_gradient_fd` — FD-vs-Jacobian directional contracts at `linearization_tol=1e-8`/`fd_tol=1e-7` with `_paired_seeds`. Solid.
- `test_boozer_derivatives_jax.py::TestBoozerHessianComposed::test_hessian_taylor_convergence` — second-order Taylor with `best_defect_tol=1e-8, min_observed_order=2.0`. Solid.
- `test_boozer_derivatives_jax.py::TestBoozerResidualCoilVJP` (3 tests) — explicit M3 keyword signature, scalar grad parity at `rtol=1e-10, atol=1e-12`. Solid.
- `test_boozersurface_jax.py::TestBoozerExactConstraintsJacobianTaylor` (2 tests) — real exact residual, Jacobian via `jax.jacfwd`, ratio=0.55 ε-ladder. Solid.
- `test_boozersurface_jax.py::TestUpstreamFactoryBoozerMatrix::test_penalty_gradient_taylor_matrix` and `test_penalty_value_and_gradient_cpu_parity_tensor_matrix` — real NCSX surfaces, multi-epsilon Taylor, CPU/JAX direct value+gradient parity. Solid.
- `test_boozersurface_jax.py::TestStellsymMaskCPUJAXParity` (3 grid configs × 3 mpol/ntor/nfp combos) — direct CPU vs JAX mask parity. Solid.
- `test_boozersurface_jax.py::TestNfpVolumeArea` (8 tests) — analytical 2π²Rr² and 4π²Rr at `_TORUS_GEOMETRY_RTOL=1e-13`. Solid.
- `test_boozersurface_jax.py::TestVJPHooks::test_ls_reduced_directional_requires_spatial_field_derivatives` — proves the dB/dX term must be present via FD-vs-symmetric-FD with active-vs-dropped comparison. Excellent.
- `test_boozersurface_jax.py::TestVJPHooks::test_add_G_current_cotangent_matches_abs_vjp_at_zero_current` — pins the abs-subgradient convention at zero current. `rtol=1e-14`. Strong.

---

## Notes on file paths and conventions

- All tests assume `_make_mock_*` helpers live in `boozersurface_jax_test_helpers.py`. Modifying `_MockVolumeLabel` (P0 #2) will impact ~70% of M4 tests.
- The `parity_ladder_tolerances("exact-well-conditioned-adjoint")` SSOT helper is correctly used in 3 places. Other lane keys (`exact-ill-conditioned-adjoint`, `branch-stable-resolve`, `fd-gradient`) are NOT exercised by any test in this bucket — see Missing Coverage.
