# Bucket 8 — Modified upstream tests (delta-only audit)

Audit base: `upstream_hss/master = 1b0cc3a96`. HEAD = `42b68f33d` on `gpu-purity-stage2-20260405`.
Date: 2026-04-25.

> Scope: only the JAX-port-specific delta vs upstream is in scope. Whitespace
> and quote normalization (single→double quote, PEP-8 wrapping, `0.5**i` →
> `0.5**i`, `1.` → `1.0`) is excluded from issue counts but still inflates the
> raw `git diff` numstat.

---

## 1. Per-file delta-only summary

| File | LOC + | LOC − | Effective new tests | Tolerance loosenings | New skips/xfails | New tautologies | Well-tightened additions | Effective regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tests/configs/test_zoo.py | 8 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| tests/configs/test_zoo_mock_quasr.py | 25 | 22 | 0 | 0 | 0 | 0 | 0 | 0 (refactor only) |
| tests/core/test_derivative.py | 23 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| tests/core/test_dofs.py | 16 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| tests/core/test_optimizable.py | 17 | 6 | 1 (new) / 1 lost | 0 | 0 | 0 | 1 | 1 (LOST coverage in `test_call`) |
| tests/field/test_biotsavart.py | 95 | 2 | 4 | 0 | 0 | 0 | 4 (≤1e-15) | 0 |
| tests/field/test_normal_field.py | 62 | 31 | 0 | 0 | 0 (refactor of existing skipIf) | 0 | 0 | 0 |
| tests/field/test_particle.py | 15 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| tests/field/test_selffieldforces.py | 1583 | 404 | ≈11 (force-cache, etc.) | 0 explicit; large coverage cut | 1 (`test_objectives_time` env-gated) | 0 | 11 | **HIGH** — `test_force_objectives_taylor_test` matrix shrunk from 32×10 ≈ 320 cases to 2×6 = 12 |
| tests/field/test_wireframefield.py | 18 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| tests/geo/test_boozersurface.py | 594 | 174 | 9 (new sopp signature shims, guards) | 0 | 0 | 0 | 9 | 0 (rest is reformat) |
| tests/geo/test_curve.py | 1102 | 290 | 7 (CurveCWSFourier, paired_lin) | 0 | 0 | 0 | 6 | 0 (atol 1e-13/14 against CPP) |
| tests/geo/test_curve_objectives.py | 689 | 216 | 9 (chunking, JIT-cache, barriers) | 0 (barrier floor introduced) | 0 (but 3 upstream tests DELETED) | 0 | 9 | **HIGH** — see issues #1–#5 |
| tests/geo/test_strainopt.py | 105 | 76 | 0 | 0 | 0 | 0 | 0 | 0 (refactor) |
| tests/geo/test_surface.py | 49 | 18 | 1 (shapely fallback) | 0 | 1 (Shapely guard, justified) | 0 | 1 | 0 |
| tests/geo/test_surface_rzfourier.py | 9 | 5 | 0 | 0 | 0 | 0 | 2 (mpol/ntor sanity) | 0 |
| tests/geo/test_surface_taylor.py | 3 | 1 | 0 | 0 | 0 | 0 | 1 (proper module setUp/tearDown) | 0 |
| tests/mhd/test_virtual_casing.py | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 (eval→getattr cleanup) |
| tests/objectives/test_fluxobjective.py | 114 | 14 | 4 (fake fixtures + JAX guard) | 0 | 0 | **2** (fake-field self-tests) | 1 (`SquaredFluxJAX requires surface_spec`) | 2 (issues #6, #7) |
| tests/objectives/test_utilities.py | 37 | 1 | 1 + module-state fix | 0 | 0 | 0 | 1 | 0 |

**Headline regressions**

1. `tests/field/test_selffieldforces.py::test_force_objectives_taylor_test` —
   the upstream 8-deep parameter sweep (≈320 sub-cases) was collapsed into 2
   hand-picked configs (12 sub-cases). 96% coverage cut.
2. `tests/field/test_selffieldforces.py::test_objectives_time` — wrapped in
   `@unittest.skipUnless(SIMSOPT_RUN_FIELD_TIMING==1)`; default CI now skips.
3. `tests/geo/test_curve_objectives.py::test_curve_minimum_distance_taylor_test`
   — `downsample` loop (1, 2, 3) and `np.random.seed(0)` removed; threshold
   table-driven only by `CurveHelical`.
4. `tests/geo/test_curve_objectives.py::test_linking_number` — `use_jax_curve ∈
   [False, True]` parameterization deleted; only CPU branch remains.
5. `tests/geo/test_curve_objectives.py` — three upstream tests deleted outright:
   `test_arclength_variation_circle_planar`, `test_linking_number_planar`,
   `test_curve_curve_distance_empty_candidates`.
6. `tests/objectives/test_fluxobjective.py` introduces a `_FluxObjectiveFakeField`
   that overrides `B()` and `B_vjp()` with caller-supplied arrays; the new
   gradient-handles-zero-normals test only checks zeros against zeros.

---

## 2. Top issues (current HEAD)

| # | File:line (HEAD) | Description | Classification | Old vs new | Recommended tightening |
|---|---|---|---|---|---|
| 1 | tests/field/test_selffieldforces.py:1646–1854 (`test_force_objectives_taylor_test`) | Upstream sweeps `ncoils ∈ [2] × nfp ∈ [1,3] × stellsym ∈ [True] × p ∈ [2.5] × threshold ∈ [0.0, 1e-3] × reg_type ∈ {circ, rect} × downsample ∈ [1,2] × use_jax_curve ∈ [False, True] × numquadpoints ∈ [10]` against 10 objective-summation patterns — 320 Taylor cases. HEAD reduces to 2 fixed `test_configs` × 6 objectives = 12 cases. The deleted matrix is exactly the lane that exercises CPU-vs-JAX-curve parity, downsampled vs. dense, and threshold-active vs. threshold-zero gradient code paths. | HIDES-REGRESSION SKIP/XFAIL (de-facto); coverage drop | 320 → 12 sub-cases; `use_jax_curve=True/False` parametric pair lost in 5 of the 12 surviving cases | Restore the full nested loop (or use `pytest.mark.parametrize` over `(ncoils, nfp, stellsym, p, threshold, reg_name, downsample, use_jax_curve, numquadpoints)`); keep both objective-list expansion patterns (per-coil `sum(...)` AND vectorized over `coils`); fail with the original `assert err_new < 0.5 * err` rule. |
| 2 | tests/field/test_selffieldforces.py:1868–1870 | `@unittest.skipUnless(os.environ.get("SIMSOPT_RUN_FIELD_TIMING") == "1", "benchmark-only timing test")` added. Upstream ran timing unconditionally and saved `objective_runtimes_semilogy.png`. Default CI now skips. | HIDES-REGRESSION SKIP/XFAIL | Was: always-on. Now: requires opt-in env var. | Either restore default-on with a 30 s timeout guard, or split into `test_objectives_time_smoke` that runs unconditionally on a single tiny `ncoils` and asserts compile+run < threshold. |
| 3 | tests/geo/test_curve_objectives.py:639–644 (`test_curve_minimum_distance_taylor_test`) | Lost `for downsample in [1, 2, 3]` loop AND `np.random.seed(0)`. Distance threshold collapsed from constant `0.4` (covered both Helical and non-Helical) to a `0.4 if CurveHelical else 0.2` ternary, and the `subTest` no longer covers `(curvetype, rotated, downsample)`. | HIDES-REGRESSION SKIP/XFAIL + reproducibility regression | Was: 3 downsample × 6 curvetype × 2 rotated = 36 sub-cases (seeded). Now: 12 sub-cases (no seed → flaky if order changes). | Re-introduce the `downsample` loop and `np.random.seed(0)` (or per-test `default_rng(seed)`); keep the 0.4-vs-0.2 distinction explicit per `curvetype`. |
| 4 | tests/geo/test_curve_objectives.py:906–928 (`test_linking_number`) | Upstream sweeps `for use_jax_curve in [False, True]` inside `for downsample in [1, 2, 5]`. HEAD only iterates `downsample`, hard-coding the C++ curve. JAX linking-number kernel is untested. | HIDES-REGRESSION SKIP/XFAIL | 6 sub-cases → 3 sub-cases | Restore the inner `use_jax_curve` parametric loop; assert `1e-14` rtol both ways. |
| 5 | tests/geo/test_curve_objectives.py — DELETED upstream tests | `test_arclength_variation_circle_planar`, `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates` no longer exist in HEAD. The first verifies an analytic invariant (planar circle → arclength variation = 0); the second verifies a planar-knot edge case; the third verifies the `J.candidates = []` fallback path that used to compute distance directly. | HIDES-REGRESSION SKIP/XFAIL (deletion) | 3 dedicated tests removed | Restore them verbatim — these test invariants that are *not* covered by the parameterized variants that did survive. |
| 6 | tests/objectives/test_fluxobjective.py:209–217 (`test_quadratic_flux_gradient_handles_zero_normals`) | Constructs `_FluxObjectiveFakeField` with `B = zeros` and `_FluxObjectiveFakeSurface` with `normal = zeros`, then asserts `dJ() == zeros(field.local_dof_size)`. With both `B` and `n` zero, the gradient is trivially zero by construction; this is verifying `0 ⋅ 0 == 0`. The fake `B_vjp` returns whatever `dJdB` is reshaped, so the test cannot detect any bug in the real `BiotSavart.B_vjp` plumbing. | NEW WEAK ASSERTION (effectively tautological — the upstream `BiotSavart` is never exercised) | n/a (new) | Either (a) plug a real `BiotSavart` over a real coil whose `B(target_surface) == 0` by symmetry (e.g. coil + image coil) and assert `dJ` is zero to `rtol=1e-13`; or (b) keep the fake but additionally verify that `dJ` is zero *for each non-trivial component of the dofs* via a Taylor test, not just shape. |
| 7 | tests/objectives/test_fluxobjective.py:220–236 (`test_singular_local_returns_inf_and_raises_gradient_failure`, `test_singular_normalized_returns_inf_and_raises_gradient_failure`) | Both use the `_FluxObjectiveFakeField`. The fake's `B()` returns the user-supplied 0 vector and `B_vjp` is never reached because the gradient raises. These are useful smoke tests for the `ObjectiveFailure` contract but they verify nothing about the JAX backend. | NEW WEAK ASSERTION (acceptable as contract guards, but should be marked) | n/a | Add a parallel pair of tests against a real `BiotSavart` field that is *almost* singular (e.g. set `target.normal()` to a near-zero pattern) and verify that the gradient eventually overflows or that `np.isfinite(dJ).all()` false-positives are rejected before the `ObjectiveFailure`. |
| 8 | tests/core/test_optimizable.py:243–258 | Upstream `test_call` ended with five lines that mutated `opt1.set('x1', 5)` / `opt2.set('x1', 4)` and ran a follow-up `np.allclose(self.opt(), 34.0/24)` check. HEAD removed those five lines and added a brand-new `OptimizableAncestorOrderingTests::test_ancestors_keep_numeric_instance_order_past_single_digits`. The original behavioural follow-up was thrown out. | HIDES-REGRESSION SKIP/XFAIL (silent drop) | 5 lines deleted; 1 new test added | Re-add the `opt1.set/opt2.set` block to `test_call` (keep the new ordering test on top — it's good). |
| 9 | tests/geo/test_boozersurface.py:53–98 (`test_call_boozer_residual_falls_back_to_alpha_only_signature`) | Patches `boozersurface_module.sopp.boozer_residual` with a fake that raises `TypeError` on the 7-arg form to exercise the alpha-only fallback. The test mocks the C++ residual entirely; it does NOT verify that the alpha-only path produces the same numeric value as the new signature on a real surface. | NEW WEAK ASSERTION (contract-only — does not catch fall-back numerical drift) | n/a | Add an integration test where the *real* `boozer_residual` is called with both signatures (legacy alpha-only, new) on a small surface and asserts equality at `rtol=1e-13`. The current test only proves "we reach the second branch", not "the second branch is correct". |
| 10 | tests/geo/test_boozersurface.py:100–164 (`test_call_boozer_residual_ds_falls_back_to_alpha_only_signature`) and :166–227 (`_ds2`) | Same structure as #9 — checks the fallback dispatcher reaches the alpha-only branch but never validates against the real C++ kernel output. | NEW WEAK ASSERTION | n/a | Same recommendation as #9. |
| 11 | tests/objectives/test_utilities.py:71–98 (`test_quadratic_penalty_hostifies_jax_scalar_objective`) | Monkey-patches `utilities_mod._host_float_scalar` to count invocations, then asserts `calls["count"] >= 4`. Tests an internal hostification implementation detail; if the hostification is later inlined or fused, the test breaks even though the API still works. | NEW WEAK ASSERTION (tests an implementation detail) | n/a | Replace with a behavioural assertion: e.g. `penalty.J()` equals the analytic value AND `type(penalty.J())` is a Python `float` (not a `jax.Array`). Drop the call-count monkeypatch. |
| 12 | tests/core/test_derivative.py:120–127 (`test_jax_blocks_materialize_to_numpy`) | Asserts `isinstance(deriv.data[opt], jax.Array)` then that `deriv(opt, as_derivative=True).data[opt]` is `np.ndarray`. This is a CONVENTION DRIFT marker — JAX blocks are stored *as-is* in raw `data`, hostified only on the public boundary. Good test. | WELL-TIGHTENED (CONVENTION DRIFT documented) | n/a | None — this is a correct guard. |
| 13 | tests/core/test_derivative.py:129–141 (`test_mixed_numpy_and_jax_blocks_are_hostified_in_arithmetic`) | Adds a NumPy `Derivative` to a JAX `Derivative` and asserts the result is `np.ndarray` (not `jax.Array`). Documents the mixed-block hostification rule. | WELL-TIGHTENED | n/a | None. |
| 14 | tests/field/test_biotsavart.py:435–528 (4 new fieldcache tests) | New tests for the simsoptpp fieldcache compatibility layer (cold-cache fill, indexed slot tracking, legacy keys, non-canonical shapes). All assertions either compare against `dB_by_dcoilcurrents()[0]` (which is the C++ ground truth) or verify shape/status via `bs.fieldcache_get_status`. The numeric-equivalence tests use `np.testing.assert_allclose` without explicit rtol but also assert `< 1e-15` for the residual finite-difference vs. analytic gradient. | WELL-TIGHTENED | n/a | None. |
| 15 | tests/field/test_normal_field.py:23–28 | Adds a `requires_spec_runtime` decorator that skips when *either* `py_spec` is `None` *or* `spec_wrapper` is `None`. Five tests are switched from the broader `requires_py_spec` to `requires_spec_runtime`. This is a *narrowing* of the skip predicate — fewer environments will now run those tests (they now require both `py_spec` and the SPEC wrapper). | NEW SKIP (justified — the affected tests really do need the wrapper) | Was: `skipIf(py_spec is None)`. Now: `skipIf(py_spec is None or spec_wrapper is None)`. | Document in the decorator docstring that `spec_wrapper is None` typically means the SPEC F90 module is missing. Add a CI lane that ensures at least one configuration has both. |
| 16 | tests/field/test_selffieldforces.py:2503–2532 (`test_lpcurveforces_taylor_test`) | Upstream uses `h = np.ones_like(dofs)` (a fixed direction). HEAD switches to `h = rng.standard_normal(dofs.shape); h /= np.linalg.norm(h)`. This is a tightening (more general direction) but makes the test stochastic on the seeded RNG. | WELL-TIGHTENED | seed=0 → seed=0 (still seeded), but direction changed | None — the new direction is more likely to expose component-mismatched gradient bugs than `ones`. |
| 17 | tests/geo/test_curve.py:1733–1748 (`test_curvecwsfourier_h0_small_phic_regime_remains_evaluable`) | Builds two CurveCWSFourier kwargs (`G=0,H=0` and `G=1,H=0`), sets `thetas(1)=0.1`, `phic(1)=0.05`, then asserts `np.all(np.isfinite(values))` for 9 derived quantities. No equality check; only finite-ness. | NEW WEAK ASSERTION | n/a | Add a parity check against an analytic reduction (e.g. for `H=0` the phi-component should equal `2π * phic`). Right now this only catches NaN regressions. |
| 18 | tests/geo/test_curve.py:1812–1813 (`test_surface_rzfourier_paired_lin_methods_match_grid_diagonal`) | Asserts `np.testing.assert_allclose(out, diagonal, rtol=1e-12, atol=1e-12)` — matches `*_lin(out, phi, theta)` against the diagonal of the grid evaluation. This is a strong CPU-self-consistency check. | WELL-TIGHTENED | n/a | None. |
| 19 | tests/geo/test_curve.py:1827–1855 (`test_surface_rzfourier_third_paired_lin_methods_match_fd`) | Validates third-derivative `_lin` against central-difference `(eval(eps) - eval(-eps)) / (2eps)` at `rtol=1e-8, atol=1e-7`. Fixed eps=1e-6 — for double precision this is the optimal Taylor floor for second-order central difference of a third derivative, so the asserted tolerance is reasonable. | WELL-TIGHTENED | n/a | None — for this finite-difference order the tolerance is justified. |
| 20 | tests/geo/test_curve.py:1788, 1790, 1792 (`test_curvecwsfourier_matches_cpp_on_stage2_surface`) | JAX vs C++ `CurveCWSFourier` parity: `gamma` at `atol=1e-14`, `gammadash` at `atol=1e-13`, `kappa` at `atol=1e-12`. Strong same-state direct-kernel parity. | WELL-TIGHTENED | n/a | None. |
| 21 | tests/geo/test_curve_objectives.py:103–124 (`test_lp_curve_torsion_reuses_shared_jit_kernels`) | Asserts that a second `LpCurveTorsion` instance does NOT increase the cache size of the shared JIT functions. Tests a JIT-cache implementation detail. If the kernel fingerprinting changes upstream, this test fires false-positive even when the public behaviour is identical. | NEW WEAK ASSERTION (implementation-detail brittle) | n/a | Replace with a behavioural assertion: `obj1.dJ()` and `obj2.dJ()` agree to `rtol=1e-13`, and the second call's wall-clock is < 5× the first (proxy for caching). |
| 22 | tests/geo/test_curve_objectives.py:248–261 (`test_pairwise_penalty_chunking_preserves_infeasible_barrier_inf`) | Sets `SIMSOPT_JAX_PENALTY_POINT_CHUNK_SIZE=1` and asserts the chunked path returns `+inf` when the barrier is violated. Good ENV-var contract test, but does NOT also verify the dense path returns the same `+inf`. | WELL-TIGHTENED but partial | n/a | Add the dense-path comparison: assert `np.isposinf(cc_distance_barrier_pure(..., chunk_size=0))` too, so a buggy dense path can't slip through. |
| 23 | tests/geo/test_curve_objectives.py:335–376 (`test_pairwise_penalty_accepts_explicit_row_sharding`) | Constructs a JAX `Mesh` from `jax.devices()` and tests sharded vs dense agreement at `atol=1e-12`. **TAUTOLOGICAL when run on single-device CPU** (mesh is degenerate), but adds value on multi-device CI. | TAUTOLOGICAL on single-device runs | n/a | Add an `@pytest.mark.skipif(len(jax.devices()) == 1, reason="...")` so single-device CI doesn't pretend to test sharding. Or: add an explicit oracle (compute the rowwise distances in NumPy and compare). |
| 24 | tests/geo/test_curve_objectives.py:548–578 (`subtest_curve_curvature_barrier_taylor_test`) | The `_assert_barrier_taylor_progress` helper introduces a "barrier Taylor floor" of `5e-12`: when `err > 10 * 5e-12`, require `err_new < 0.6 * err`; else require `err_new <= max(1.05 * err, 5e-12)`. This is a *deliberately loosened* tolerance for the barrier penalty's near-floor regime. | WELL-TIGHTENED (justified — the barrier penalty has a finite floor at machine epsilon × condition number; 5e-12 is correct for the chosen `1e-2 * randn` step) | n/a (new test) | Document in the helper docstring why `5e-12` is the floor; reference the parity-ladder lane `derivative-heavy: rtol=1e-8` so future maintainers don't tighten this back to `0.55 *` blindly. |
| 25 | tests/geo/test_boozersurface.py:264–273 (`test_run_code_rejects_G_none_with_free_currents`) | Constructs a stub BoozerSurface via `__new__` (bypassing `__init__`) and asserts `run_code(0.1, None)` raises `ValueError, "fixed coil currents when G=None"`. Good guard test, but the stub is so synthetic that it doesn't catch a bug where the wrong `biotsavart` attribute name is checked first (only the new `_make_free_current_biotsavart("coils")` path is verified — the same test exists for `"_coils"` separately). | WELL-TIGHTENED | n/a | None — companion test on line 275 covers the legacy `_coils` attribute path. |

---

## 3. Tolerance loosening regressions

**Result: zero explicit tolerance loosenings detected** in any of the 20 audited
files. Every numeric assertion in the JAX-port delta either matches the
upstream tolerance verbatim or tightens it (e.g. `test_lpcurveforces_taylor_test`
moves to a normalized random direction).

The only "loosening-shaped" change is the new `_assert_barrier_taylor_progress`
helper (issue #24) which introduces a `5e-12` floor for the barrier penalties
— this is a *new* test for a *new* objective class, not a relaxation of an
existing assertion, and the floor is mathematically justified.

That said, the *coverage* loosening (issues #1–#5) is much more severe than any
plausible numeric loosening would have been: it removes whole axes (
`use_jax_curve`, `downsample`, `numquadpoints`, `nfp`) from the parameter sweep
on the very objectives the JAX port actually rewrote.

---

## 4. Skip / xfail regression audit

| # | File:line | Predicate | Hides regression? | Recommendation |
|---|---|---|---|---|
| S1 | tests/field/test_selffieldforces.py:1868–1870 | `@unittest.skipUnless(os.environ.get("SIMSOPT_RUN_FIELD_TIMING") == "1")` on `test_objectives_time` | YES — upstream ran this unconditionally; default CI now skips, masking timing regressions in `LpCurveForce`/`LpCurveTorque`/`SquaredMeanForce`/`SquaredMeanTorque`. | Restore default-on with `ncoils=2` only; gate the larger sweep on the env var. |
| S2 | tests/geo/test_surface.py:578 | `@unittest.skipIf(LineString is None, ...)` on `test_is_self_intersecting_shapely_fallback` | NO — the test only exercises the Shapely fallback path that requires `shapely`. Justified skip predicate. | None — this is correct. |
| S3 | tests/field/test_normal_field.py:24–28 | `requires_spec_runtime = unittest.skipIf(not _SPEC_RUNTIME_AVAILABLE, ...)` applied to 5 previously `requires_py_spec`-only tests | LOW RISK — the old predicate was already a heuristic skip; the new one is stricter. | Add an integration CI lane that has both `py_spec` and the SPEC wrapper installed so these 5 tests actually run. |
| S4 | (none other) | — | — | — |

The total new skip count is 1 hard regression (S1) plus 1 justified addition (S2)
plus 5 narrowed predicates (S3). No `xfail` markers were added in any audited
file.

---

## 5. Tightening playbook (P0)

Order by ROI. Each item is a single-PR-sized restoration.

1. **Restore `test_force_objectives_taylor_test` parameter sweep**
   (`tests/field/test_selffieldforces.py`, issue #1). Re-introduce the 9-deep
   nested loop, or convert it to `pytest.mark.parametrize` over the same axes.
   This is the single largest test-coverage regression in the audited bucket and
   it covers exactly the JAX-port code paths (`use_jax_curve`, `downsample`).
2. **Re-enable `test_objectives_time` by default** (issue #2). Gate only the
   `ncoils=8` row on `SIMSOPT_RUN_FIELD_TIMING=1`; keep the smallest `ncoils=2`
   row default-on with a soft assertion (compile time < 60 s). Default-off
   timing tests do not catch timing regressions.
3. **Restore `downsample` loop and `np.random.seed(0)`** in
   `subtest_curve_minimum_distance_taylor_test` (issue #3). The seed removal
   alone introduces order-dependent flakiness; the `downsample` removal removes
   coverage of the JAX chunked-distance kernel.
4. **Restore `use_jax_curve` parameterization in `test_linking_number`**
   (issue #4). The JAX linking-number kernel is currently unverified.
5. **Un-delete `test_arclength_variation_circle_planar`,
   `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates`**
   in `tests/geo/test_curve_objectives.py` (issue #5). These are
   invariant/edge-case guards that the surviving parameterized variants do not
   cover.
6. **Re-add the parent-mutation follow-up to `test_call`** in
   `tests/core/test_optimizable.py` (issue #8). Five lines were silently
   dropped when the new `OptimizableAncestorOrderingTests` class was added.
7. **Replace `_FluxObjectiveFakeField` self-tests with real-`BiotSavart`
   equivalents** in `tests/objectives/test_fluxobjective.py` (issues #6–#7).
   The current zero-against-zero tests cannot detect bugs in the real
   `BiotSavart.B_vjp` plumbing.
8. **Replace JIT-cache-size assertions with behavioural ones** in
   `test_lp_curve_torsion_reuses_shared_jit_kernels` and
   `test_framed_curve_twist_reuses_shared_jit_kernels`
   (`tests/geo/test_curve_objectives.py`, issue #21). Cache-size internals
   are brittle; `obj.J()` parity + wall-clock heuristic is the testable
   contract.
9. **Replace `_host_float_scalar` call-count monkeypatch with type assertion**
   in `tests/objectives/test_utilities.py` (issue #11). Implementation-detail
   counts will break under future inlining without indicating real regressions.
10. **Add real-kernel oracle to `test_call_boozer_residual*`** trio
    (`tests/geo/test_boozersurface.py`, issues #9–#10). The current tests prove
    "we hit the second branch" but never confirm the second branch is
    numerically correct.

---

### Methodology note

Every bullet above was verified by reading the diff against
`upstream_hss/master` at `1b0cc3a96` and the corresponding section of HEAD on
`gpu-purity-stage2-20260405` at `42b68f33d`. Whitespace and quote-style
reformatting (which dominates the raw `git diff --numstat` LOC counts in
`test_boozersurface.py`, `test_curve.py`, `test_curve_objectives.py`,
`test_selffieldforces.py`, `test_normal_field.py`) was excluded from issue
classification.
