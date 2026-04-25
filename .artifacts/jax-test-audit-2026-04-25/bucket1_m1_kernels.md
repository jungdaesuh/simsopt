# Bucket 1 — M1 Pure JAX Kernel Tests Audit

Date: 2026-04-25
Branch: gpu-purity-stage2-20260405
Auditor: max-effort test-quality auditor

Files in scope:
- `tests/field/test_biotsavart_jax.py` (1002 lines)
- `tests/field/test_biotsavart_jax_parity.py` (833 lines)
- `tests/geo/test_surface_fourier_jax.py` (400 lines)
- `tests/geo/test_surface_rzfourier_jax.py` (783 lines)
- `tests/objectives/test_integral_bdotn_jax.py` (374 lines)

Source modules judged:
- `src/simsopt/field/biotsavart_jax.py` (shim → `src/simsopt/jax_core/biotsavart.py`)
- `src/simsopt/geo/surface_fourier_jax.py` (XYZTensor + XYZ Fourier evaluators)
- `src/simsopt/jax_core/surface_rzfourier.py` (RZ Fourier evaluators)
- `src/simsopt/objectives/integral_bdotn_jax.py` (3-definition surface integral)

Tolerance reference (PARITY_LADDER_TOLERANCES):
- `direct_kernel`: rtol=1e-10, atol=1e-12 (same-state C++ parity)
- `derivative_heavy`: first_derivative_rtol=1e-8, atol=1e-10; second_derivative_rtol=1e-6, atol=1e-8
- `_REDUCTION_ACCEPTANCE_TIERS["biotsavart_chunked_dense"]`: cpu (1e-12, 1e-14), gpu (1e-12, 1e-13)

---

## 1. Per-file summary table

Counts include parametrized expansions (e.g., one `@pytest.mark.parametrize` with 3 values is counted as 3).

| file | total tests | tautological | loose | weak | meaningless | mocked | well-tightened | priority |
|---|---|---|---|---|---|---|---|---|
| tests/field/test_biotsavart_jax.py | ~24 | 1 | 5 | 2 | 3 | 0 | 13 | **P0** |
| tests/field/test_biotsavart_jax_parity.py | ~32 | 1 | 1 | 5 | 0 | 0 | 25 | P1 |
| tests/geo/test_surface_fourier_jax.py | ~13 | 1 (orthogonality by construction) | 2 | 1 | 0 | 0 | 9 | **P0** |
| tests/geo/test_surface_rzfourier_jax.py | ~24 | 4 (parity to JAX gradient via JAX gradient) | 3 | 4 (HLO heuristic gates) | 4 (HLO heuristic gates) | 0 | 9 | **P0** |
| tests/objectives/test_integral_bdotn_jax.py | ~25 | 0 | 1 | 1 | 0 | 0 | 23 | P1 |

Notes:
- "Total" is approximate because parametrization explodes some tests; weighting reflects effort.
- The `test_biotsavart_jax_parity.py::TestCurveTypeParametrization` is essentially a smoke pass over four curve types, but it has at least one assertion of physical content (linearity in I, divergence-free), so it is mostly well-tightened with the obvious gaps called out in §2.

---

## 2. Top issues — ranked worst-first

| # | file:line | test name | classification | one-line evidence (quote) | concrete tightening recommendation |
|---|---|---|---|---|---|
| 1 | tests/geo/test_surface_rzfourier_jax.py:619-651 | `_assert_surface_jacobian_parity` (called by stellsym + non-stellsym variants) | **TAUTOLOGICAL** for the `_jax(dofs)` arm | `np.testing.assert_allclose(normal_jacobian, np.asarray(surface.dnormal_by_dcoeff_jax(dofs)), rtol=1e-12, atol=1e-12)` and similarly for `dunitnormal_by_dcoeff_jax` | The rtol=1e-12 arm compares a JAX call against the same JAX implementation routed through the surface object — both call `surface_rz_fourier_dnormal_from_dofs`. The only meaningful arm is the C++ comparison at rtol=1e-9. Drop the JAX-vs-JAX 1e-12 assertion or replace with an FD assertion at h=1e-6, rtol=1e-8 against a direction-perturbed normal computed via `surface_rz_fourier_normal_from_spec`. |
| 2 | tests/geo/test_surface_rzfourier_jax.py:662-696 | `_assert_area_volume_gradient_parity` (called by stellsym + non-stellsym variants) | **TAUTOLOGICAL** for the `_jax(dofs)` arm | `np.testing.assert_allclose(np.asarray(surface.darea_by_dcoeff_jax(dofs)), area_grad, rtol=1e-12, atol=1e-12)` where `area_grad = jax.grad(surface_rz_fourier_area_from_dofs)(dofs)` | Same pattern: `darea_by_dcoeff_jax` and the local `jax.grad` both go through identical JAX code. Keep only the C++-oracle comparison at rtol=1e-9 (lines 673-684) and the JAX-vs-FD comparison via a small step-size sweep. |
| 3 | tests/geo/test_surface_fourier_jax.py:135-147 | `test_normal_orthogonality` | **TAUTOLOGICAL** | `n = surface_normal(...)`; `dot1 = jnp.sum(n * gd1, axis=-1); np.testing.assert_allclose(dot1, 0.0, atol=1e-12)` | `surface_normal` is implemented as `jnp.cross(gd1, gd2)`. By the algebraic identity `(a × b) · a = (a × b) · b = 0`, this assertion is a tautology of cross-product arithmetic and cannot fail unless XLA breaks linear algebra. Replace with: assert `n` matches an analytic torus normal (R, r torus has known closed form `n = (R+r cos θ) (cos θ cos φ, cos θ sin φ, sin θ) · 2π · 2π`) at rtol=1e-13. |
| 4 | tests/field/test_biotsavart_jax.py:929-961 | `test_grouped_biot_savart_jit_accepts_forced_point_sharding` | **WEAK ASSERTION** | `assert result.shape == (4, 3); assert jnp.all(jnp.isfinite(result))` | This advertises sharding parity but only checks shape + finiteness. Add `np.testing.assert_allclose(np.asarray(result), np.asarray(biot_savart_B(points, gammas, gammadashs, currents)), rtol=1e-12, atol=1e-14)` so the shard-vs-replicate path is value-checked, not just survived. |
| 5 | tests/field/test_biotsavart_jax.py:891-927 | `test_grouped_biot_savart_accepts_explicit_point_sharding` | **MEANINGLESS PHYSICS** for the sharding-type assertion | `assert isinstance(grouped_B.sharding, NamedSharding)` after a value parity check | The value parity is fine. The sharding-class assertion is an implementation-detail check; if the API ever returns `PositionalSharding` or `SingleDeviceSharding` for a 1-device CPU host the test fails for no physical reason. Replace with `assert grouped_B.sharding == points.sharding` or drop the isinstance check. |
| 6 | tests/field/test_biotsavart_jax.py:383-401 | `test_div_B_zero` | **POTENTIALLY LOOSE on tightness, otherwise OK** | `np.testing.assert_allclose(np.array(div_B), 0.0, atol=1e-14)` | The autodiff symbolically yields div = 0 at every quadrature point (the integrand is itself div-free pointwise via `(r̂ × dl)/r²` ⇒ `∂_j (cross/r³)_j = 0` analytically), so `1e-14` should be safely achieved. The test passes — but **this is the only div-B check with on-axis evaluation; add an additional point set of N=20 random off-axis points with `atol=1e-13` to broaden coverage**. (See also missing-coverage entry on tighter divergence-free check.) |
| 7 | tests/field/test_biotsavart_jax.py:424-452 | `test_dB_dX_finite_difference` | **LOOSE TOLERANCE** | `eps=1e-5; ... fd_rel_tol = 1e-8; fd_abs_tol = 5e-11` for centred FD on float64 | Central FD has truncation error `O(eps²) ≈ 1e-10` and round-off `~ε_mach/eps ≈ 1e-11`. The combined achievable accuracy at `eps=1e-5` is `~1e-10`. Setting `rtol=1e-8` is one-to-two orders of magnitude looser than achievable. Either tighten to `rtol=1e-9, atol=1e-10` or replace this single-eps check with a Taylor-test convergence loop (matching the style of `_assert_point_perturbation_taylor_convergence` already in the parity file). The convergence-loop variant catches tangent-space errors that single-eps masks. |
| 8 | tests/geo/test_surface_fourier_jax.py:101-133 | `test_gammadash1_finite_difference`, `test_gammadash2_finite_difference` | **LOOSE TOLERANCE** | `eps = 1e-7; ... rtol=1e-5, atol=1e-10` for centred FD | At `eps=1e-7`, central FD truncation is `~eps² · (f''') = 1e-14`, but round-off is `~ε_mach/eps = 1e-9`. Best achievable is `~1e-9`; the test sets `rtol=1e-5` which is 10000× looser than necessary. Either set `eps=1e-5, rtol=1e-9, atol=1e-12` or use a Taylor-test sweep. The current tolerance hides off-by-one indexing errors that would still pass. |
| 9 | tests/field/test_biotsavart_jax.py:455-471 | `test_multiple_coils` | **TAUTOLOGICAL (additivity by construction)** | `B_total = biot_savart_B(points, gammas, gammadashs, currents); B1 = ...; B2 = ...; np.testing.assert_allclose(np.array(B_total), np.array(B1 + B2), atol=1e-14)` | Biot-Savart in the kernel is `μ₀/(4π) * Σ_c I_c · ∫_c integrand`. The grouped path is literally `B(coils=[c1,c2]) = B(coils=[c1]) + B(coils=[c2])` by Σ algebra inside `_one_point_dense`. The kernel is linear in coils so this passes by construction. Keep as a smoke test or replace with: assert `B_total` matches an analytic two-loop superposition (e.g., Helmholtz coil pair: `B_z(0) = 8μ₀I/(5√5 R)` for separation `R`) at `rtol=1e-12`. |
| 10 | tests/field/test_biotsavart_jax.py:520-540 | `test_backend_cache_invalidation_clears_kernel_cache` | **MEANINGLESS PHYSICS** | `assert core_bs._make_kernel.cache_info().currsize == 0` | This tests an implementation detail (lru_cache size). It does not validate that physics is preserved across cache invalidation. Add a value parity assertion: capture `B` before invalidation, invalidate, recompute, assert identical to atol=0 (bitwise) so a future contract change that reseeds RNGs or alters reduction order cannot pass silently. |
| 11 | tests/field/test_biotsavart_jax.py:542-593 | `test_B_vjp_rebuilds_when_tuning_changes_in_process` | **MEANINGLESS PHYSICS for the cache-currsize gate; mostly OK for the parity gate** | `assert core_bs._make_B_vjp_kernel.cache_info().currsize == 1` then `... == 2` | Lines 587-593 do a real value parity at atol=1e-14. Keep that. The cache-count assertions are implementation churn risk; consider relaxing to `>= 1` and `>= 2` so they can survive a JIT lifecycle change without coupling test to internal cache counter semantics. |
| 12 | tests/geo/test_surface_rzfourier_jax.py:441-449 | `test_surface_rzfourier_geometry_allows_strict_transfer_guard` | **WEAK ASSERTION** | `with jax.transfer_guard("disallow"): gamma, xphi, xtheta = geometry_fn(spec); gamma.block_until_ready()` (no value check) | Tests only that no host transfer occurs, not that values are correct. Add a value parity vs `surface.gamma()` at rtol=1e-12 inside the same with-block to ensure the strict-transfer code path produces the same physics, not just runs. |
| 13 | tests/geo/test_surface_rzfourier_jax.py:452-477 | `test_surface_rzfourier_fused_geometry_reduces_hlo_work` | **MEANINGLESS PHYSICS / IMPLEMENTATION DETAIL** | `assert fused_lowered_stats["cosine"] < scalar_lowered_stats["cosine"]` | This is a regression gate on XLA HLO trig-op count using regex on rendered text. Fragile to JAX/XLA version, brittle to fusion-pass changes, and provides zero coverage of correctness. Mark `@pytest.mark.brittle_perf_gate` or move into a benchmarks/ guard suite, not the kernel test bucket. If kept, add a value parity check (fused vs scalar geometry must match to `atol=1e-14`) so regressions in correctness can't be hidden by an HLO win. |
| 14 | tests/geo/test_surface_rzfourier_jax.py:479-517 | `test_surface_rzfourier_scalar_gamma_hlo_stays_single_output` | **MEANINGLESS PHYSICS / IMPLEMENTATION DETAIL** | `assert gamma_stats["line_count"] < geometry_stats["line_count"]` | Same critique as #13 — pure HLO heuristic. No correctness tested. Either move out of M1 kernel suite or add value parity. |
| 15 | tests/geo/test_surface_rzfourier_jax.py:520-554 | `test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package` | **MEANINGLESS PHYSICS** | `assert payload["comparison"]["hlo_gate_passed"] is True` | Subprocess-spawning test that asserts a JSON-payload boolean produced by a benchmark probe. Doesn't exercise the kernel, only the benchmark CLI's success report. Move to `benchmarks/` test suite. |
| 16 | tests/geo/test_surface_rzfourier_jax.py:580-591 | `test_surface_rzfourier_unitnormal_degenerate_surface_stays_finite` | **MEANINGLESS PHYSICS** | `unitnormal = ...(spec); assert np.all(np.isfinite(unitnormal)); np.testing.assert_array_equal(unitnormal, np.zeros_like(unitnormal))` | Tests an arbitrary failure-handling convention (zeros for degenerate surface). Reasonable as a contract test, but does not validate that the contract is the right one. Add: also check that `surface_rz_fourier_normal_from_spec` returns zeros (not NaN) for the same input and that `surface_rz_fourier_area_from_spec` returns 0.0 (the integral of zero magnitude normals). |
| 17 | tests/field/test_biotsavart_jax.py:341-362 | `test_on_axis_field` | **WELL-TIGHTENED but `nquad=256` is overkill for `rtol=1e-12`**; symmetry assertion uses `atol=1e-14` | `B_analytical = MU0 * I / (2.0 * R); np.testing.assert_allclose(float(B[0, 2]), B_analytical, rtol=1e-12)` | Physically correct, well-tightened. Keep. **One observation**: with `nquad=256` and a smooth circular loop, the trapezoid quadrature on a periodic integrand is super-exponentially convergent — the test would pass at `nquad=32` to the same precision. Lower `nquad` to `64` to keep test fast (8× speedup) without compromising the gate. |
| 18 | tests/field/test_biotsavart_jax_parity.py:124-156 | `_assert_point_perturbation_taylor_convergence` | **WELL-TIGHTENED** | Multi-eps loop `for i in range(5, 10): eps = 0.5**i; ...; assert new_err < 0.55 * err` (forward FD, expected ratio 0.5) | Genuine Taylor convergence gate. Choice of 0.55 leaves modest slack for round-off. Suggest tightening to 0.51 or use an FD-error envelope `eps · |f''|/2 + ε_mach·|f|/eps` for adaptive bounds. |
| 19 | tests/field/test_biotsavart_jax_parity.py:158-202 | `_assert_second_derivative_taylor_convergence` | **WELL-TIGHTENED** | Central FD on first derivative; `assert new_err < 0.30 * err` (expected ratio 0.25) | Correct ratio for central FD. Keep. |
| 20 | tests/field/test_biotsavart_jax.py:484-497 | `test_B_parity_ncsx` | **WELL-TIGHTENED** | `np.testing.assert_allclose(np.array(B_jax), B_ref, rtol=1e-10)` | Direct C++ oracle parity at `direct_kernel` lane tolerance. Exemplary. |
| 21 | tests/field/test_biotsavart_jax.py:499-517 | `test_dB_by_dX_parity_ncsx` | **WELL-TIGHTENED** | `rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"]` (= 1e-8) | Per the contract this is the right lane. Keep. |
| 22 | tests/field/test_biotsavart_jax_parity.py:269-291 | `test_B_is_curl_A` | **WELL-TIGHTENED but tolerance could be tighter** | `np.testing.assert_allclose(curl_A, B, atol=1e-14)` | Excellent invariant test (Maxwell). Tolerance reasonable. Note: B has magnitude ~μ₀I/R ~ 1e-2 at the eval points, so `rtol=1e-12` would be a tighter alternative; current `atol=1e-14` is roughly equivalent at this magnitude. |
| 23 | tests/field/test_biotsavart_jax_parity.py:341-361 | `test_dB_dX_symmetric_and_divergence_free` | **WELL-TIGHTENED** | `assert abs(dB_idx[0,0]+dB_idx[1,1]+dB_idx[2,2]) < 1e-14`; `np.testing.assert_allclose(dB_idx, dB_idx.T, atol=1e-12)` | Both Maxwell invariants checked. Symmetry tolerance `1e-12` is appropriate — `dB` magnitude at near-coil points can be O(1) so `1e-12` is ~12 digits of agreement. |
| 24 | tests/field/test_biotsavart_jax_parity.py:805-817 | `test_B_linearity_in_current` | **WELL-TIGHTENED** | `assert err < 1e-15` for `B_full - I·B_unit` per curve type | Bit-exact linearity in I — Biot-Savart is exactly linear in current, so this is a strong contract test. Keep. |
| 25 | tests/field/test_biotsavart_jax_parity.py:749-764 | `test_gamma_nontrivial` (one of the parametrized smokes) | **WEAK ASSERTION** | `assert gamma_extent.max() > 0.01; assert np.min(gammadash_norms) > 1e-10` | Order-of-magnitude smoke. Acceptable as a parametrized presence-of-curve check, but **add at least one Cartesian-coordinate parity check vs a known reference per curve type** (e.g., for `CurveXYZFourier` with the upstream get_curve DOF layout, the analytic gamma is known). |
| 26 | tests/field/test_biotsavart_jax_parity.py:805-829 | `test_B_cross_type_consistency` | **WEAK ASSERTION** | `assert diff > 1e-10, f"{curvetype}: B unchanged after DOF perturbation"` | This only asserts that DOFs are wired through. A bug that perturbs B in the wrong direction (e.g., wrong sign) would still pass. Replace with an FD-of-B-vs-DOF Taylor test using `jax.grad(curve_spec_with_dofs(...) ∘ biot_savart_B)`. |
| 27 | tests/field/test_biotsavart_jax.py:836-889 | `test_point_chunked_B_A_dB_dA_match_dense_reference` | **LOOSE / SELF-PARITY** | `np.testing.assert_allclose(np.asarray(B), np.asarray(dense_B), atol=1e-14)` where `dense_B = jax.vmap(_one_point_dense)(...)` | This is a chunked-vs-unchunked self parity. It is a useful internal regression gate but is NOT a parity test against a true oracle. The assertion `atol=1e-14` is correct for the same numerical recipe — but the test name suggests external parity. Rename to `test_point_chunked_matches_dense_self` and/or add an additional comparison vs a cross-implementation oracle (C++ or analytic-loop) at `rtol=1e-10`. |
| 28 | tests/objectives/test_integral_bdotn_jax.py:39-47 | `_make_test_data` synthetic fixture used by all parametric parity tests | **SOMEWHAT SYNTHETIC** | `B = rng.randn(nphi, ntheta, 3) * 0.1 + np.array([0, 0, 1.0]); ... normal = rng.randn(...) * 0.5` | The B-field, target, and normals are not from a self-consistent (B,n,A) triple. Parity vs NumPy is unaffected — the function is purely algebraic — so this is a valid contract test of the objective formula. **Add at least one test that drives the integrand from a real Biot-Savart B over a real torus surface normal, asserting that integral_BdotN("quadratic flux") for a *closed* surface around a *coil-free* region equals zero to machine precision** (this is the most physically meaningful invariant: ∫ B·n dA = 0 for any closed surface enclosing no current). |
| 29 | tests/objectives/test_integral_bdotn_jax.py:132-154 | `test_zero_when_B_tangential` | **WELL-TIGHTENED** | `np.testing.assert_allclose(J, 0.0, atol=1e-25)` | Strong invariant: B in tangent plane → BdotN=0. The `atol=1e-25` is overly strict (effectively atol=0 on float64), but the test passes by analytic identity (B is constructed orthogonal to n). Keep. |
| 30 | tests/objectives/test_integral_bdotn_jax.py:217-251 | `test_strict_oracle_scalar_reduction_matches_high_precision_reference` | **WELL-TIGHTENED** | `reference = 0.5 * math.fsum(value*value for value in amplitudes) / amplitudes.size; np.testing.assert_allclose(strict_oracle_value, reference, rtol=1e-15, atol=1e-4); assert abs(strict_oracle - reference) < abs(default - reference)` | Excellent: uses Kahan-equivalent `math.fsum` as the gold standard and asserts strict_oracle is closer than default. Keep. |

---

## 3. Missing coverage

Physical/mathematical invariants that should be tested in this bucket but aren't:

- **`integral_BdotN("quadratic flux") == 0` for a divergence-free B over a closed surface enclosing no current.** This is the single most natural physical check on the entire objective. None of the test files instantiate a real B-from-coils + real surface normal pair. Add a torus surface (use `_make_simple_torus_coeffs`) with coils displaced outside the torus and assert `integral_BdotN < 1e-12`. (Currently `tests/objectives/test_integral_bdotn_jax.py` is purely algebraic against synthetic arrays.)
- **Surface area parity vs analytic torus formula.** `surface_area(normal)` is exported; for a torus with R=1, r=0.1 the analytic surface area is `4π² R r = 4π²·0.1 ≈ 3.948`. No test asserts this. Add: for the `_make_simple_torus_coeffs` torus with `nphi=ntheta=128`, `surface_area(normal)` must equal `4π² R r * nfp` (where the nfp factor accounts for the field-period quadrature reduction in `surface_fourier_jax.surface_area`) at `rtol=1e-10`.
- **Surface volume parity vs analytic torus formula.** For the same torus, `V = 2π² R r²`. `surface_volume` is exported and untested in the file. Add the analytic check at `rtol=1e-10`.
- **Stellsym DOF round-trip on surface_xyzfourier (XYZTensorFourier).** `tests/geo/test_surface_fourier_jax.py` has C++ parity tests for `dgamma_by_dcoeff` but does not cover stellsym=True for XYZTensor; the parametrized `test_coefficient_derivatives_match_cpp` uses `stellsym=False` only. Add `stellsym=True` parameter to that parametrize.
- **Long-wire limit for `biot_savart_B`.** A straight infinite wire has `B = μ₀ I / (2π r)` at perpendicular distance `r`. None of the tests check this limit (they only check the circular-loop on-axis case). Add a test using a long thin solenoid or a long straight segment approximation; the wire limit catches singular-integrand handling at small `r`.
- **Vector potential `A` gauge consistency.** `biot_savart_A` is tested only via `B = curl(A)`. No direct A-vs-analytic check (e.g., circular-loop `A_φ = μ₀ I R / (4π) · (2/k) [(1 - k²/2) K(k) - E(k)] / √(R r)`). Adds confidence A is right beyond the curl-of-A oracle (which only fixes B = ∇×A up to a gauge; A could still be wrong by a gradient).
- **Gauge invariance of A (∇·A is well-defined).** Not strictly required, but Biot-Savart A is in the Coulomb gauge (∇·A = 0). Add: `assert |Tr(dA_dX) at evaluation point| < 1e-12`.
- **`dB/dX` symmetry off-axis at multiple points (vacuum constraint).** `tests/field/test_biotsavart_jax.py::test_div_B_zero` checks div B but **not** symmetry of `dB/dX` (vacuum = curl B = 0 at points away from current ⇒ `dB[i,j] = dB[j,i]`). The parity file's `test_dB_dX_symmetric_and_divergence_free` covers this for the Fourier coil at idx∈{0,16}. Move that idiom into the analytical test file too at the off-axis points already used in `test_div_B_zero`.
- **Stellarator-symmetric DOF unpacking is consistent with the analytical reflection (φ,θ) → (-φ,-θ).** The `stellsym_scatter_indices` function has a non-trivial convention (cos-cos+sin-sin for x; cos-sin+sin-cos for y,z). No test directly verifies that a randomly chosen stellsym surface satisfies `gamma(-φ, -θ) = (gamma(φ,θ).x, -gamma(φ,θ).y, -gamma(φ,θ).z)`. Add this symmetry check at `atol=1e-13` in `tests/geo/test_surface_fourier_jax.py`.
- **Surface periodicity in `(φ, θ)`.** `surface_gamma(0, θ) == surface_gamma(1/nfp, θ)` and `surface_gamma(φ, 0) == surface_gamma(φ, 1)` should hold to machine precision. No periodicity test exists.
- **`grouped_biot_savart_B` parity vs flat `biot_savart_B` for matching coil sets.** `test_grouped_biot_savart_accepts_explicit_point_sharding` does this on the side, but as a stand-alone invariant it is not exposed. Add `test_grouped_equals_flat_biot_savart` with `atol=1e-14`.
- **`biot_savart_B_vjp` C++ parity** when simsoptpp is available. The B and dB/dX are checked against C++; the VJP is only checked self-consistency (Taylor test). simsoptpp's `BiotSavart` exposes `B_vjp`-equivalent through coil current/curve gradient pullbacks; a parity gate against that would close a gap.
- **`biot_savart_d2B_by_dXdX` C++ parity** when simsoptpp is available. The `test_d2B_dXdX_symmetric` is good, but the second-derivative tier of PARITY_LADDER_TOLERANCES (`second_derivative_rtol=1e-6`) is currently uncovered for d²B. simsoptpp provides `d2B_by_dXdX` so add the parity gate.
- **Mixed-quadrature group equivalence to flat call.** `test_mixed_quad_gradient_fd` validates the gradient through the grouped path against FD, but no test asserts that the grouped FORWARD value equals the flat-coil-list forward value when both have the same total set of coils. Add a forward-value parity to complement the gradient FD.
- **`integral_BdotN("normalized")` invariance under uniform B scaling.** If you multiply B and target uniformly by α, `J_normalized` is invariant (numerator scales α², denominator scales α²). No invariance test exists.
- **`integral_BdotN("quadratic flux")` quadratic scaling under B scaling.** Multiplying B by α should multiply J by α². Easy to add as a one-line invariant test.
- **Integration consistency: `integral_BdotN` recovered from `residual_BdotN`.** `J = 0.5 · ‖r‖²` where `r = residual_BdotN(...)`. Worth one assertion: `0.5 * float(jnp.dot(r, r))` should equal `integral_BdotN` to machine precision in `default` mode.

---

## 4. Tightening playbook (P0)

The 8 most impactful, low-effort changes:

1. **`tests/geo/test_surface_rzfourier_jax.py:619-651, 662-696` — drop tautological JAX-vs-JAX assertions.** Remove the `_jax(dofs)` rtol=1e-12 arms inside `_assert_surface_jacobian_parity` and `_assert_area_volume_gradient_parity`; they compare JAX path to the same JAX path routed through the surface object. Keep only the C++ parity arms (`rtol=1e-9, atol=1e-9`) and add an FD oracle:
   ```python
   from numpy.testing import assert_allclose
   eps = 1e-5
   for k in range(min(5, dofs.size)):
       perturbed = dofs.at[k].add(eps)
       n_plus = surface_rz_fourier_normal_from_spec(spec_with(perturbed))
       perturbed_m = dofs.at[k].add(-eps)
       n_minus = surface_rz_fourier_normal_from_spec(spec_with(perturbed_m))
       fd_col = (host_array(n_plus) - host_array(n_minus)) / (2*eps)
       assert_allclose(normal_jacobian[..., k], fd_col, rtol=1e-7, atol=1e-9)
   ```

2. **`tests/geo/test_surface_fourier_jax.py:135-147` — replace `test_normal_orthogonality` with an analytic-torus normal check.** Cross-product orthogonality is a tautology. New test:
   ```python
   # For (R,r) torus with R=1, r=0.1, nfp=1: |n| = (R + r cos θ) · r at every (φ,θ)
   args = (self.phis, self.thetas, ...)
   n = surface_normal(*args)
   norm_n = jnp.sqrt(jnp.sum(n*n, axis=-1))
   theta_2d = 2*np.pi * np.array(self.thetas)[None, :]
   expected = (self.R + self.r * np.cos(theta_2d)) * self.r * (2*np.pi)**2 / 1.0  # absorb the 2π factors from gammadash chain rule
   np.testing.assert_allclose(np.array(norm_n), np.broadcast_to(expected, norm_n.shape), rtol=1e-13)
   ```

3. **`tests/geo/test_surface_fourier_jax.py:101-133` — tighten the FD tolerances.** Change `eps=1e-7, rtol=1e-5, atol=1e-10` → `eps=1e-5, rtol=1e-9, atol=1e-12`. Central FD with `eps=1e-5` is good to ~1e-10 on float64; the existing tolerance gives a 10000× slack that masks indexing bugs.

4. **`tests/field/test_biotsavart_jax.py:424-452` — replace single-eps FD test with Taylor convergence loop.** The parity file already has `_assert_point_perturbation_taylor_convergence`. Reuse it from the analytical test instead of the looser `rtol=1e-8, atol=5e-11` single-eps check. Mechanical change:
   ```python
   from .test_biotsavart_jax_parity import _assert_point_perturbation_taylor_convergence
   _assert_point_perturbation_taylor_convergence(
       biot_savart_B, biot_savart_dB_by_dX,
       jnp.array([[0.4, 0.1, 0.05]]), gammas, gammadashs, currents, idx=0,
   )
   ```

5. **`tests/field/test_biotsavart_jax.py:929-961` — add value-parity to `test_grouped_biot_savart_jit_accepts_forced_point_sharding`.** Replace the bare-shape/finite check with:
   ```python
   reference = biot_savart_B(points, gammas, gammadashs, currents)
   np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=1e-12, atol=1e-14)
   ```

6. **`tests/objectives/test_integral_bdotn_jax.py` — add the divergence-theorem invariant test.** Concrete:
   ```python
   def test_quadratic_flux_zero_for_external_coils_over_closed_surface(self):
       from src.simsopt.geo.surface_fourier_jax import (
           surface_gamma, surface_normal,
       )
       # torus (R=1, r=0.1) at origin
       phis = jnp.linspace(0, 1.0, 64, endpoint=False)
       thetas = jnp.linspace(0, 1.0, 64, endpoint=False)
       xc = jnp.zeros((3, 1)); yc = jnp.zeros((3, 1)); zc = jnp.zeros((3, 1))
       xc = xc.at[0,0].set(1.0); xc = xc.at[1,0].set(0.1); zc = zc.at[2,0].set(0.1)
       gamma = surface_gamma(phis, thetas, xc, yc, zc, 1, 0, 1)
       normal = surface_normal(phis, thetas, xc, yc, zc, 1, 0, 1)
       # coil at z = 5 (well outside torus)
       coil_g = jnp.array([[[5.0*np.cos(t), 5.0*np.sin(t), 5.0] for t in np.linspace(0,2*np.pi,64,endpoint=False)]])
       coil_gd = jnp.array([[[-5.0*2*np.pi*np.sin(t), 5.0*2*np.pi*np.cos(t), 0.0] for t in np.linspace(0,2*np.pi,64,endpoint=False)]])
       points = gamma.reshape(-1, 3)
       B = biot_savart_B(points, coil_g, coil_gd, jnp.array([1e6])).reshape(gamma.shape)
       J = float(integral_BdotN(B, jnp.zeros(gamma.shape[:2]), normal, "quadratic flux"))
       # ∫ B·n = 0 for divergence-free B over a closed surface enclosing no current.
       # quadratic flux is integral_of_(B·n)² > 0 in general; the surface integral of B·n itself must be zero.
       # Use residual instead:
       from simsopt.objectives.integral_bdotn_jax import residual_BdotN
       r = residual_BdotN(B, jnp.zeros(gamma.shape[:2]), normal, "quadratic flux")
       # The signed sum of weighted (B·n) should be ~0 because Σ B·n |n|/(N) ≈ ∫ B·n dA = 0.
       signed_sum = float(jnp.sum(r * jnp.sign(jnp.sum(normal*B, axis=-1)).reshape(-1)))  # crude oracle
       # Better: compute the unweighted-signed integral directly:
       BdotN = jnp.sum(B * normal/jnp.linalg.norm(normal, axis=-1, keepdims=True), axis=-1)
       norm_n = jnp.linalg.norm(normal, axis=-1)
       gauss_integral = float(jnp.sum(BdotN * norm_n) / (gamma.shape[0]*gamma.shape[1]))
       assert abs(gauss_integral) < 1e-10
   ```
   This is the canonical Gauss-law check missing from the bucket.

7. **`tests/geo/test_surface_fourier_jax.py` — add analytic surface area & volume tests.** Two new tests using `_make_simple_torus_coeffs(R=1.0, r=0.1)`:
   ```python
   from simsopt.geo.surface_fourier_jax import surface_area, surface_volume
   nphi, ntheta = 128, 128
   gamma = surface_gamma(phis, thetas, xc, yc, zc, mpol, ntor, nfp)
   normal = surface_normal(phis, thetas, xc, yc, zc, mpol, ntor, nfp)
   # NB: surface_area divides by (nphi*ntheta) and the nfp factor cancels with quadrature step
   assert_allclose(float(surface_area(normal)), 4 * np.pi**2 * R * r, rtol=1e-10)
   assert_allclose(float(surface_volume(gamma, normal)), 2 * np.pi**2 * R * r**2, rtol=1e-10)
   ```
   Without these, the area/volume helpers in `surface_fourier_jax.py` are completely uncovered by direct invariant tests.

8. **`tests/geo/test_surface_rzfourier_jax.py:452-554` — quarantine HLO-text gating tests.** Move `test_surface_rzfourier_fused_geometry_reduces_hlo_work`, `test_surface_rzfourier_scalar_gamma_hlo_stays_single_output`, and `test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package` out of the M1 kernel suite into a `tests/benchmarks/` or `tests/perf_gates/` directory and tag with `@pytest.mark.brittle_perf_gate`. They contribute zero correctness coverage and break under JAX/XLA upgrades. If kept in the main suite, append a value-parity assertion to each so a regression in correctness can't be hidden by an HLO win.

---

## Closing notes

The bucket is largely well-engineered — `tests/field/test_biotsavart_jax_parity.py` in particular contains genuine Taylor-test gates for first and second spatial derivatives, current-linearity bit-exactness, and Maxwell invariants (curl A = B, div B = 0, dB/dX symmetric in vacuum). Those tests are exemplary.

The systemic weaknesses concentrate in two places:

1. **`tests/geo/test_surface_rzfourier_jax.py`** has multiple tautological JAX-vs-JAX assertions (lines 619-696) and several HLO-text heuristic gates that don't validate physics. These should be the first cleanup target.
2. **`tests/geo/test_surface_fourier_jax.py`** uses overly-loose FD tolerances (rtol=1e-5 on float64 central FD!) and a tautological `test_normal_orthogonality`. Two of the three exported reductions (`surface_area`, `surface_volume`) are not directly invariant-tested at all.

The integral_BdotN test file is structurally sound but missing the most important physical invariant (Gauss law on a closed surface).
