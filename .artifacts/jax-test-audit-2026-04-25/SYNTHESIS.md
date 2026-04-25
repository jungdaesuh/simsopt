# JAX Test-Quality Audit — Synthesis (2026-04-25)

Branch: `gpu-purity-stage2-20260405` (HEAD `42b68f33d`).
Source bucket reports: `.artifacts/jax-test-audit-2026-04-25/bucket{1..8}_*.md`.
Parity ladder SSOT: `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`
(lanes: `direct_kernel`, `ls_wrapper_gradient`, `derivative_heavy`,
`exact_well_conditioned_adjoint`, `exact_ill_conditioned_adjoint`,
`branch_stable_resolve`, `fd_gradient`, `gpu_runtime`).

## 1. Executive summary

- **Tests audited (approx.)**: ≈ 1,180 across the 8 buckets — bucket 1 ≈118,
  bucket 2 ≈ 306, bucket 3 ≈ 471 (incl. 285 in single_stage_example), bucket 4
  ≈ 51, bucket 5 ≈ 388, bucket 6 ≈ 53, bucket 7 ≈ 36, bucket 8 ≈ 53 effective
  new tests. **Total findings logged across buckets: ≈ 195.**
- **Top-3 systemic risks**:
  1. **GPU "production proof" suite cannot prove a GPU was used.** All 31
     tests in `tests/test_hf_production_gpu_proof.py` route through
     `tests/subprocess/hf_production_gpu_fake_runner.py`, the proof bundle has
     no `default_backend`/`devices`/`xla_flags`/`jaxlib.cuda_versions` field,
     and `benchmarks/stage2_e2e_comparison.py` is missing the
     `require_requested_platform_runtime` guard that `single_stage_init_parity`
     already uses. The 2026-04-20 Runpod cubin incident would still be GREEN.
  2. **`_MockVolumeLabel.J() ≡ 0.0` silently nulls the constraint contribution
     in ~70 % of M4 BoozerSurfaceJAX tests.** Lives in
     `tests/geo/boozersurface_jax_test_helpers.py:521-612`. Half the penalty
     physics (target_label − J) is constant in those tests, so any sign or
     magnitude bug in the volume/area constraint gradient cannot fail.
  3. **FD-vs-IFT escape hatches hide sign errors and below-2× Taylor decay.**
     `_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55` (above the proper 0.25 for symmetric
     central FD), `_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3` of 5 directions
     (majority gate), `rel < 1e-3 OR abs < 1e-8` (OR-escape) — all in
     `tests/integration/test_single_stage_jax_cpu_reference.py:1979,4213,5792`.
     A wrong-sign IFT term on a single direction still passes.
- **Bucket-by-bucket triage**:
  - **Solid**: bucket 1 parity file (`test_biotsavart_jax_parity.py`); bucket 2
    M3 derivative tests (`test_boozer_derivatives_jax.py`), `TestVJPHooks`
    directional FD, `TestNfpVolumeArea`, `TestStellsymMaskCPUJAXParity`;
    bucket 4 Stage 2 CPU-vs-JAX parity floor (B/dB/J/dJ at 1e-10..1e-12),
    JIT-closure contract test (line 2121), strict-mode mixed-quad negative
    test (line 2180); bucket 5 backend snapshot/restore conftest; bucket 8
    `test_curve.py` CurveCWSFourier C++ parity at 1e-13/14.
  - **Fragile**: bucket 1 surface_rzfourier JAX-vs-JAX taut tests + HLO-text
    gates; bucket 2 `boozersurface_jax_test_helpers._MockVolumeLabel`-driven
    suite; bucket 3 `test_surface_objectives_jax.py` 32/68 monkeypatch-only
    tests; bucket 5 `test_jax_import_smoke.py` ~70 wrappers that only assert
    `rc == 0`; bucket 7 `test_accessibility.py` 5/6 cache-routing-only tests;
    bucket 8 `test_curve_objectives.py` and `test_selffieldforces.py`
    coverage-axis collapse vs upstream.
  - **Theatre**: bucket 6 entire `test_hf_production_gpu_proof.py` (0/31
    exercise GPU); bucket 5 `test_field_cache_hot_path_benchmark.py`
    (TAUTOLOGICAL printf-format pinning); bucket 3
    `test_iotas_jax_exact_well_conditioned_gradient_matches_dense_projection`
    (toy 3×3 self-test labelled as `exact-well-conditioned-adjoint` lane).
- **What I would trust on this branch right now**: Stage 2 fixed-surface flux
  parity (CPU-vs-JAX), Biot-Savart B + dB/dX direct-kernel parity at 1e-10,
  M3 composed Boozer derivatives (Taylor + check_grads), `BoozerSurfaceJAX`
  exact-well-conditioned operator-vs-PLU adjoint parity on the *synthetic*
  diagonal fixture, the `derivative_heavy` lane wherever it is named, and the
  Stage 2 mixed-quadrature value/grad path. **What I would NOT trust**: any
  GPU claim (no real-GPU lane runs), any IotasJAX/NQSR resolve-FD test that
  passes only because of the 2-of-3 / OR-escape gates, any
  `BoozerSurfaceJAX` test using `_MockVolumeLabel` for constraint physics,
  any `*_reuses_shared_jit_kernels` accessibility test, the donation
  `donate_argnums=(0,)` invariant (no test asserts buffer deletion), and the
  three TAUTOLOGICAL kernel "contract" tests in
  `test_fluxobjective_jax_parity.py:211-263`.

## 2. Cross-cutting themes

| # | Theme | Definition | Observed in | Severity | Root-cause hypothesis |
|---|---|---|---|---|---|
| T1 | **Tautological JAX-vs-JAX "parity"** | Both sides of the assertion route through the same JAX kernel/path | bucket 1 (test_surface_rzfourier_jax.py:619-696, test_surface_fourier_jax.py:135-147), bucket 3 (test_surface_objectives_jax.py:1728, test_single_stage_jax_cpu_reference.py:2326), bucket 4 (test_fluxobjective_jax_parity.py:211, 223, 253) | P0 | Tests written against the implementation, not against an oracle (CPP, analytic, or `math.fsum`) |
| T2 | **Mock contamination zeroes the contribution under test** | A fixture stub returns 0 / identity for the very quantity the test is supposed to validate | bucket 2 (`_MockVolumeLabel.J()→0.0`, `_successful_exact_newton_result.jacobian = I`), bucket 3 (`test_real_fixture_ondevice_parity_and_wrapper_gradients` patched callbacks), bucket 5 (~14 `fake_run_python_script` wrappers in `test_benchmark_helpers.py`), bucket 6 (entire fake runner) | P0 | Mock helpers were written for plumbing tests then reused as physics fixtures without re-stubbing the load-bearing parts |
| T3 | **OR-of-relative-and-absolute escape hatches** | Acceptance is `rel < X OR abs < Y` where Y is below the FD floor, so any near-zero gradient passes | bucket 3 (test_single_stage_jax_cpu_reference.py:4213, 5792, 1979) | P0 | "Just make it pass on the noisy fixture" instead of using the `fd_gradient` lane (`directional_fd_rtol=1e-5, directional_fd_atol=1e-7`) |
| T4 | **Smoke-only acceptance for proof / GPU lanes** | Exit-code-only or fake-payload-accepted for tests advertised as proof | bucket 5 (~75 subprocess wrappers in `test_jax_import_smoke.py` assert only `rc == 0`), bucket 6 (entire `test_hf_production_gpu_proof.py`), bucket 5 (`test_runpod_single_stage_continuation.py:374` shell-string match) | P0 | No JSON-payload sentinel pattern propagated from the gold-standard `test_single_stage_surface_reprojection_probe` (line 847) |
| T5 | **Coverage axis collapse vs upstream** | Parametric loops dropped or tests deleted outright | bucket 8 (`test_force_objectives_taylor_test`: 320 → 12 sub-cases; `test_curve_minimum_distance_taylor_test`: lost downsample loop + RNG seed; `test_linking_number`: lost `use_jax_curve` parametrization; 3 upstream tests deleted) | P0 | "Trim CI runtime" without a parity audit; the trimmed axes are exactly the JAX-port-rewritten code paths |
| T6 | **Missing physical invariants** | ∇·B=0 off-axis, ∮B·n dA=0 over closed surface, gauge invariance, nfp rotational symmetry, analytic torus area/volume | bucket 1 (no Gauss-law test; no analytic torus area at `surface_area`), bucket 4 (no nfp rotational symmetry of B; no closed-surface Gauss-law gate); bucket 2 (no Boozer residual scaling-property test) | P1 | Tests written against the implementation; a "free" oracle (analytic torus, divergence theorem) was never plumbed in |
| T7 | **Implementation-detail testing** | Asserts on cache `currsize`, lru_cache size, JIT-cache hits, HLO line counts, `device_get` call counts | bucket 1 (HLO regex tests at test_surface_rzfourier_jax.py:452-554), bucket 5 (test_field_cache_hot_path_benchmark.py:23-49 printf-format), bucket 7 (5/6 accessibility tests pin `_cache_size() == 1`), bucket 3 (test_single_stage_jax.py:115, 138 count `device_put` calls) | P1 | Confused brittle XLA/JIT internals with public contract |
| T8 | **Order-dependent / leaky-fixture state** | Test names encode an order that pytest does not guarantee under `-p random`/`-x` | bucket 5 (`test_backend_state_guard_sequence_01..04`, `test_backend_module_guard_sequence_01..02` mutating module-level dict), bucket 7 (`tests/geo/conftest.py` snapshot at conftest-import time) | P1 | No `pytest-ordering` / `@pytest.mark.dependency`; manual `_NN` numbering used as a substitute |
| T9 | **Donation / device residency NOT verified** | Tests use `donate_argnums=(0,)` but never assert buffer deletion; tests assert `transfer_guard("disallow")` does not raise but never assert that an obvious offending op DOES raise | bucket 5 (test_biotsavart_donation_probe.py:75-95 — `_fresh_points()` per call so donation invariant cannot fail by construction); bucket 1 (test_surface_rzfourier_jax.py:441-449 — no value check inside the `disallow` block); bucket 6 (no device-residency in proof bundle) | P0 | Contract was written, no mechanism exists to verify it from the test layer |
| T10 | **Adjoint contract drift (well-conditioned vs ill-conditioned, dense vs operator)** | Tests labelled with the wrong lane key; tautological synthetic A=I fixtures masquerade as exact-well-conditioned coverage; ill-conditioned adjoint vector parity asserted instead of residual-only | bucket 2 (test_boozersurface_jax.py:3946 fixture A=diag, NOT a real Boozer Jacobian; line 3831 patched solver returns rhs unchanged then asserts solved≈rhs); bucket 3 (test_surface_objectives_jax.py:1728 toy 3×3); bucket 5 (`exact-ill-conditioned-adjoint` lane is unexercised at vector parity, which is correct, but no test exists that constructs a *real* ill-conditioned exact problem) | P0 | The lane SSOT lives in `validation_ladder_contract.py` but tests cite the lane name without enforcing the lane's full contract (residual_rel_tol, vector_parity_required, requires_well_conditioned_jacobian) |
| T11 | **VJP/optimizer signature drift not pinned** | M3 (`(adjoint, gamma=, xphi=, ...)`) vs M4 LS/exact (`(lm, booz_surf, iota, G)`) coexist; no `inspect.signature` regression guard | bucket 2 (no test catches a future regression to the CPU-style `(lm, booz_surf)` signature; only `test_run_code_rejects_bad_group_vjp_signature` checks `vjp_groups` arity) | P1 | Convention is documented in CLAUDE.md but no automated guard exists |
| T12 | **Loose FD tolerances on float64 central FD** | `eps=1e-7, rtol=1e-5` where `eps=1e-5, rtol=1e-9` is achievable | bucket 1 (test_surface_fourier_jax.py:101-133), bucket 2 (test_boozer_residual_jax.py:300, 412), bucket 7 (test_simsoptpp_compat.py:60: `rtol=1e-6` for 3rd-derivative FD where `rtol=1e-9` is the truncation/rounding optimum) | P1 | Tolerances picked to "pass on first try" rather than derived from `O(eps²) + ε_mach/eps` |

## 3. Top-25 P0 fixes (global ranking)

Order: severity × confidence × leverage. Effort: XS≤30 min, S≤2 h, M≤1 day,
L>1 day.

| # | file:line | Fix | Why it matters | Effort |
|---|---|---|---|---|
| 1 | tests/test_hf_production_gpu_proof.py:1-end + benchmarks/stage2_e2e_comparison.py:46-59 | Rename to `test_hf_production_gpu_proof_shell.py` AND add `require_requested_platform_runtime` to the Stage 2 probe AND add `bundle_provenance{default_backend,devices,jaxlib_cuda_versions,xla_flags}` to every payload AND add a `tests/test_hf_production_gpu_proof_real.py` lane gated by `jax.default_backend()=="gpu"` | The 2026-04-20 Runpod incident would still pass; "GPU proof" is currently a launcher plumbing suite | L |
| 2 | tests/geo/boozersurface_jax_test_helpers.py:521-612 (`_MockVolumeLabel.J()`) | Make `_MockVolumeLabel.J()` actually compute `surface_volume(gamma, normal)` (or use `Volume(surface)` from `label_constraints_jax`) | ~70 % of M4 BoozerSurfaceJAX tests have a constant constraint contribution and cannot fail on a constraint-gradient sign/magnitude bug | S |
| 3 | tests/integration/test_single_stage_jax_cpu_reference.py:1979 + 4213 + 5792 | Drop `or abs_err < 1e-8`; require `len(stable_samples)==len(direction_samples)` (not 2-of-3 / 3-of-5); set `_REAL_RESOLVE_FD_TAYLOR_RATE=0.4`; map all three to the `fd_gradient` lane (`directional_fd_rtol=1e-5, atol=1e-7`) | Removes the systemic gate that hides wrong-sign IFT gradients on individual directions | S |
| 4 | tests/test_biotsavart_donation_probe.py:75-95 + benchmarks/biotsavart_donation_probe.py:231,237,244 | Add a probe case that retains the donated `points` array (no `_fresh_points()`) and asserts `points.is_deleted()` (or `jnp.asarray(points)` raises) | `donate_argnums=(0,)` could silently regress to `()` and every existing test still passes | S |
| 5 | tests/objectives/test_fluxobjective_jax_parity.py:211, 223, 253 | Replace the three `_flux_kernel_value_and_grad`-vs-`_flux_kernel_value_and_grad` "contract" tests with CPU `SquaredFlux` parity OR analytic-zero algebra; pin `SquaredFluxJAX.dJ()` raises `ObjectiveFailure` for the singular case | Three tests advertised as parity prove only kernel self-consistency; the singular-grad case is misleading because the public adapter rejects the gradient | S |
| 6 | tests/integration/test_single_stage_jax_cpu_reference.py:5662 (IotasJAX branch in `test_gradient_wrappers_operator_status_on_exact_state`) | Add `assert np.linalg.norm(adjoint_state.apply_transpose(adj) - dJ_ds) / (np.linalg.norm(dJ_ds)+1e-30) <= 1e-10` | The `exact-ill-conditioned-adjoint` lane requires `residual_rel_tol=1e-10`; today only finite/non-zero is checked | XS |
| 7 | tests/geo/test_surface_objectives_jax.py:1728 | Replace toy 3×3 oracle with a real `BoozerSurfaceJAX(boozer_type='exact')` fixture; materialize `J_dense` by applying `solve_transpose_with_status` to identity columns and compare against `scipy.linalg.lu_solve` | The test currently solves the same A with the same algorithm on both sides; the lane gate must consume a real exact Boozer fixture | M |
| 8 | tests/integration/test_single_stage_jax_cpu_reference.py:5859-5917 (`test_adjoint_fraction_diagnostic`) | Either promote to `assert adjoint_fraction > 0.05` OR delete and fold into the resolve-FD test as a logger.info | Currently a `>0` ceremony that any non-trivial gradient passes; pretends to be a test | XS |
| 9 | tests/integration/test_single_stage_jax_cpu_reference.py:4925 (`test_outer_opt_decreases_objective`) | Change `assert j_final <= j0 + 1e-12` to `assert j_final < j0 - 1e-6 * abs(j0)` AND `assert result.nit > 0` | `+1e-12` slack on a 3-iteration L-BFGS-B accepts rounding noise as success | XS |
| 10 | tests/geo/boozersurface_jax_test_helpers.py + tests/geo/test_boozersurface_jax.py:5443-5469 + entire `TestBoozerSurfaceJAXExactPath` cluster (~14 tests using `_patched_exact_newton_result`) | Add ONE end-to-end exact-Newton test on a real torus (no `_patched_exact_newton_result`) that asserts `‖res["residual"]‖ < 1e-10`, `‖J^T r‖ < 1e-10`, `res["jacobian_materialized"] is True` | Today's exact-path tests validate result-dict plumbing but never run a real exact solve; a regression in `_select_exact_residual_fn`/`_make_exact_residual` would slip | M |
| 11 | tests/field/test_selffieldforces.py:1646-1854 (`test_force_objectives_taylor_test`) | Restore the 9-deep nested loop (or convert to `pytest.mark.parametrize`); use the original `assert err_new < 0.5 * err` rule | Single largest test-coverage regression vs upstream: 320 → 12 sub-cases; the deleted axes (`use_jax_curve`, `downsample`, `numquadpoints`, `nfp`) are exactly the JAX-port-rewritten code paths | M |
| 12 | tests/geo/test_curve_objectives.py:639-644, 906-928 + 3 deleted tests | Restore `downsample` loop and `np.random.seed(0)` in `test_curve_minimum_distance_taylor_test`; restore `for use_jax_curve in [False, True]` in `test_linking_number`; un-delete `test_arclength_variation_circle_planar`, `test_linking_number_planar`, `test_curve_curve_distance_empty_candidates` | The JAX linking-number kernel is currently unverified; the deleted tests cover analytic invariants the surviving parameterized variants do not | S |
| 13 | tests/geo/test_surface_rzfourier_jax.py:619-696 (`_assert_surface_jacobian_parity`, `_assert_area_volume_gradient_parity`) | Drop the `rtol=1e-12` JAX-vs-JAX arms; keep only the C++ oracle arm; add an FD oracle at `eps=1e-5, rtol=1e-7, atol=1e-9` | Two helpers used by stellsym + non-stellsym variants assert JAX path against the same JAX path routed through the surface object | XS |
| 14 | benchmarks/stage2_e2e_comparison.py + run_production_gpu_proof.sh | Add `require_requested_platform_runtime` AND extend the bundle aggregator schema to require `cpu_oracle_value`/`gpu_value`/`value_rtol`/`gradient_rtol` AND reject the bundle when any rtol exceeds the parity-ladder contract | The Stage 2 lane silently falls back to CPU on a CUDA-less host today; the bundle has no per-rung parity field | M |
| 15 | tests/subprocess/jax_runtime_cases.py + tests/test_jax_import_smoke.py:519-526 (subprocess wrappers) | Have `_run_compile_count_case` print a JSON payload `{"compile_count": handler.count}`; the wrapper parses and asserts `payload["compile_count"] == 1` | The compile-count invariant is invisible at the pytest layer; a future case that silently no-ops still passes | S |
| 16 | tests/integration/test_stage2_jax.py:1036 (`test_singular_zero_current_objectives_boundary_is_documented`) | Pin `np.isposinf(cpu_j) == np.isposinf(jax_j)` (and same for `nan`); assert `isinstance(jax_j, float)` | Today only `np.isfinite(...)` is checked; CPU and JAX could disagree on `+inf` vs `-inf` vs `nan` and pass | XS |
| 17 | tests/geo/test_surface_fourier_jax.py:135-147 (`test_normal_orthogonality`) + missing analytic torus area/volume | Replace `(a×b)·a == 0` (algebraic identity) with analytic torus normal-magnitude test; ADD analytic surface area test against `4π²·R·r` and volume against `2π²·R·r²` | The orthogonality test is a tautology of cross-product arithmetic; `surface_area`/`surface_volume` exports are untested by direct invariant | S |
| 18 | tests/test_backend.py:1772-1830 (`test_backend_state_guard_sequence_01..04`, `test_backend_module_guard_sequence_01..02`) | Convert the 4-test + 2-test sequences into single tests whose body drives the full mutation/restore lifecycle; eliminate the module-level `_backend_module_guard_reloaded` dict | Test ordering is enforced only by alphabetical naming; under `pytest -p random` or `-x` the assertion becomes vacuous | S |
| 19 | tests/integration/conftest.py:54-66 (`_patch_meta_path_finder`) | Replace silent `return False` with `pytest.skip("scikit-build editable finder not installed; integration tests need it")` at module load | If the upstream finder is renamed, integration tests would silently re-route to the foreign package without anyone noticing | XS |
| 20 | tests/objectives/test_integral_bdotn_jax.py | Add the divergence-theorem invariant test: torus + external coil → `|Σ B·n · |n| / (Nphi·Ntheta)| < 1e-10` | The most natural physical check is missing; current tests are purely algebraic against synthetic arrays | S |
| 21 | tests/integration/test_stage2_jax.py + tests/objectives/test_fluxobjective_jax_parity.py | Add `test_squaredfluxjax_dJ_taylor_test` mirroring upstream `check_taylor_test` across all three definitions; add the chunked-VJP large-point-cloud parity test (companion to `test_chunked_grouped_paths_match_cpu_on_large_point_cloud` which only checks B) | A coupled CPU+JAX bug that affects both paths identically would slip; chunked VJP regression invisible | S |
| 22 | tests/geo/test_accessibility.py:89-248 (5 `*_reuses_shared_jit_kernels` tests) | Replace cache-size-only assertions with FD parity at `h=1e-6, rtol=1e-6` for J/dJ; keep cache-size as a secondary check | 5 of 6 accessibility tests assert only `_cache_size() == 1`; zero numeric J/dJ/ddJ coverage despite easy FD checks being available | M |
| 23 | tests/test_field_cache_hot_path_benchmark.py:8-66 (3 tests) | Replace `assert command[:5] == [...]`, `assert "15.00x" in summary`, `assert args.warmup == 0` with one test that compiles + runs the benchmark with `--iterations 1 --samples 1 --warmup 0` and parses JSON | Current tests are TAUTOLOGICAL printf-format pinning; nothing exercises actual cache behaviour | S |
| 24 | tests/test_run_code_benchmark_common.py:7-9 (`_force_x64` autouse) | Add a counter-test (`def test_resolver_rejects_float32_runtime`) that drops the autouse override | None of the 4 tests prove the resolver behaves correctly when x64 is *not* enabled | XS |
| 25 | tests/geo/test_boozersurface_jax.py:5536-5554 (`test_ls_vjp_returns_correct_shapes`) + add `inspect.signature` regression for `_boozer_ls_coil_vjp`/`_boozer_exact_coil_vjp` | Replace shape-only assertion with value parity vs `_boozer_ls_coil_vjp(lm, booz_surf, iota, G)` at `direct-kernel rtol=1e-12`; pin both M3 and M4 signatures | Current shape-only test would pass with a transposed wrong-axis cotangent; no automated guard catches signature drift back to the CPU `(lm, booz_surf)` form | S |

## 4. P1 / P2 backlog (grouped by theme)

### Loose FD tolerances (T12)
- `tests/geo/test_simsoptpp_compat.py:60` — tighten 3rd-derivative FD from `rtol=1e-6` to `rtol=1e-9`.
- `tests/geo/test_surface_fourier_jax.py:101-133` — `eps=1e-7, rtol=1e-5` → `eps=1e-5, rtol=1e-9`.
- `tests/geo/test_boozer_residual_jax.py:300-331, 412-428` — tighten `test_grad_iota`/`test_grad_G`/`test_hessian_matches_grad_fd` to `rtol=1e-8` / `rtol=1e-6` with ε-ladder.
- `tests/field/test_biotsavart_jax.py:424-452` — replace single-eps FD with the parity file's `_assert_point_perturbation_taylor_convergence`.

### Tautological JAX-vs-JAX (T1)
- `tests/integration/test_single_stage_jax_cpu_reference.py:2326` (`test_value_path_matches_residual_helper_not_penalty_objective`) — replace internal-helper "expected" with CPU `BoozerResidual.J()`.
- `tests/integration/test_stage2_jax.py:2228` (`test_strict_mode_uses_spec_native_forward_path`) — add CPU value parity, not just `B().shape == points.shape`.
- `tests/field/test_biotsavart_jax.py:455-471` (`test_multiple_coils`) — superposition is true by Σ algebra in `_one_point_dense`; replace with Helmholtz-pair analytic test.

### Implementation-detail testing (T7)
- `tests/integration/test_single_stage_jax.py:115, 138` — `device_get`/`device_put` call-count assertions; replace with type/dtype assertions on the public boundary.
- `tests/objectives/test_utilities.py:71-98` (`test_quadratic_penalty_hostifies_jax_scalar_objective`) — replace `_host_float_scalar` call-count monkeypatch with `type(penalty.J()) is float`.
- `tests/geo/test_curve_objectives.py:103-124` (`test_lp_curve_torsion_reuses_shared_jit_kernels`) — replace cache fingerprint assertion with `obj1.dJ()` vs `obj2.dJ()` parity at `rtol=1e-13`.

### Weak smoke/sharding "tests"
- `tests/field/test_biotsavart_jax.py:891-961` — `isinstance(grouped_B.sharding, NamedSharding)` is brittle; require value parity vs `biot_savart_B(...)` at `rtol=1e-12`.
- `tests/geo/test_curve_objectives.py:335-376` — `test_pairwise_penalty_accepts_explicit_row_sharding` is TAUTOLOGICAL on single-device CPU; add `@pytest.mark.skipif(len(jax.devices())==1, ...)` or a NumPy oracle.
- `tests/geo/test_curve.py:1733-1748` (`test_curvecwsfourier_h0_small_phic_regime_remains_evaluable`) — only finite-ness; add analytic check (for H=0 the phi-component should equal `2π·phic`).

### Mock contamination (T2) beyond M4
- `tests/objectives/test_fluxobjective.py:209-236` — three new singular tests use `_FluxObjectiveFakeField` with constant `B`; replace at least one with a real `BiotSavart` over a coil/image-coil pair.
- `tests/test_benchmark_helpers.py:1616-2351` — ~14 `fake_run_python_script` monkeypatches; pair each with at least one integration counterpart.

### Adjoint contract drift (T10)
- Add `inspect.signature` regression for `_boozer_ls_coil_vjp` and `_boozer_exact_coil_vjp` (CLAUDE.md documents the M4 signature `(lm, booz_surf, iota, G[, weight_inv_modB])` — no automated guard).
- Add an actually-ill-conditioned exact-path test that asserts `failure_category="scaling_limit"` OR operator residual `≤1e-10` and makes NO vector-parity claim. The `exact_ill_conditioned_adjoint` lane is currently unexercised against a real ill-conditioned configuration.

### Coverage-axis collapse (T5) beyond P0
- `tests/core/test_optimizable.py:243-258` — re-add the 5-line `opt1.set/opt2.set` follow-up to `test_call` that was silently dropped.
- `tests/field/test_selffieldforces.py:1868-1870` — restore `test_objectives_time` default-on with `ncoils=2`; gate only the `ncoils=8` row on `SIMSOPT_RUN_FIELD_TIMING=1`.

### Missing physical invariants (T6)
- nfp rotational symmetry of `BiotSavartJAX.B`: 10-line test that would have caught the historical Y/Z stellsym DOF convention bug noted in CLAUDE.md.
- Stellsym round-trip on `surface_xyzfourier` (`tests/geo/test_surface_fourier_jax.py` covers stellsym=False only).
- `gauss_integral` of `B·n` over closed torus surface in both bucket 1 (`test_integral_bdotn_jax.py`) and bucket 4 (`test_stage2_jax.py`).

### Subprocess wrappers (T4) beyond P0
- ~75 wrappers in `test_jax_import_smoke.py:247-300` only assert `rc == 0`; add a `OK:<case_name>` sentinel + `assert "OK:..." in stdout`.
- `tests/subprocess/jax_runtime_cases.py:36` — wrap module-level `_prefer_local_simsopt_source_tree()` and top-level `import jax/jnp/numpy` in try/except so a single import error does not collapse ~50 subprocess tests.

### Order-dependent / leaky-fixture state (T8)
- `tests/geo/conftest.py` — snapshot taken at conftest *import* time; document the dependency or move into a session-scoped fixture.
- `tests/conftest.py:434-463` — hard-coded path strings in `pytest_collection_modifyitems`; warn if a known file no longer exists.

### SSOT drift
- Move `_BACKEND_RUNTIME_ENV_VARS` into `simsopt.backend.runtime` as a frozen tuple; today the same set is partially repeated in `tests/conftest.py` (28 vars), `tests/test_jax_import_smoke.py:_BACKEND_SELECTOR_ENV_VARS` (18 vars), and `benchmarks/validation_ladder_common.py:repo_pythonpath_env`.

### Donation / device residency (T9) beyond P0
- `tests/geo/test_surface_rzfourier_jax.py:441-449` — add value parity inside the `disallow` block.
- `TestRealFixtureGpuM5Parity` — add a determinism gate (`np.testing.assert_array_equal(grad_a, grad_b)` from two same-state runs) to validate that the deterministic XLA flag is actually applied.

## 5. Suite-level recommendations

### Lane discipline

The SSOT in `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`
defines 8 lanes: `direct_kernel`, `ls_wrapper_gradient`, `derivative_heavy`,
`exact_well_conditioned_adjoint`, `exact_ill_conditioned_adjoint`,
`branch_stable_resolve`, `fd_gradient`, `gpu_runtime`.

Mismatches found:
- `test_iotas_jax_exact_well_conditioned_gradient_matches_dense_projection`
  (`test_surface_objectives_jax.py:1728`) cites the
  `exact_well_conditioned_adjoint` lane but uses a toy 3×3 self-test that
  satisfies neither `requires_well_conditioned_jacobian` nor
  `vector_parity_required` against an independent oracle.
- `test_get_adjoint_runtime_state_exact_jacobian_uses_host_tolerance_boundary`
  (`test_boozersurface_jax.py:3831`) cites
  `exact_well_conditioned_adjoint` rtol=1e-6 but the patched solver returns
  `rhs` unchanged → the assertion `solved≈rhs` is `X==X` and validates plumbing
  only.
- `exact_ill_conditioned_adjoint` lane is **unexercised at the contract level**
  (residual-only, vector parity NOT required). No test in bucket 2 or 3
  constructs an actually ill-conditioned exact problem and asserts
  `residual_rel_tol=1e-10` without claiming vector parity.
- `branch_stable_resolve` and `fd_gradient` are correctly named in the gold
  `TestCompositeGradientPipeline.test_branch_stable_composite_gradient_fd_and_descent`
  (line 4263) but the resolve-FD helpers at lines 1979/5792 use hand-tuned
  floors (`_REAL_RESOLVE_FD_TAYLOR_RATE=0.55`, `MIN_STABLE_SAMPLES=3`) instead
  of the lane's `directional_fd_rtol=1e-5, directional_fd_atol=1e-7`.
- `gpu_runtime` lane (`requires_x64`, `requires_fixed_seed`,
  `requires_runtime_metadata`) is **wholly unexercised** by the proof bucket
  (bucket 6) since no test touches a real GPU.

### New helpers worth adding
- `taylor_test_with_floor(lane="derivative_heavy", ratio=0.5, ε_ladder=...)` in
  `tests/_helpers/parity.py` that consumes a parity-ladder lane label and
  derives the FD tolerance + ratio bound from the SSOT instead of hand-tuning
  in each test.
- `assert_device_residency(arr, "gpu", index=0)` in `tests/_helpers/jax.py`
  that fails clearly when an array is on the wrong device. Used by the new
  GPU-real lane.
- `assert_buffer_donated(arr)` that asserts `arr.is_deleted()` (or the JAX 0.9.2
  equivalent) — eliminates the donation theatre.
- `forbid_fake_runner_in_proof_lane` decorator that reads
  `bundle["bundle_provenance"]["runner"]` and refuses to mark the test green
  unless `SIMSOPT_FAKE_GPU=1` is set.
- `assert_no_jax_vs_jax_parity(actual, expected, source)` that errors when
  `source` argues the assertion is a parity test but both sides came from the
  same JAX module (use `inspect.getmodule`).

### CI changes
- Add a real-GPU lane to `.github/workflows/jax_smoke.yml` gated by
  `jax.default_backend()=="gpu"` with `pytest.mark.skipif(strict=True)` — a
  missing GPU on the GPU lane must be RED, not skipped.
- Run pytest with `-p random` (e.g. `pytest-randomly`) at least nightly to
  catch the order-dependent `test_backend_*_sequence_*` and module-level state
  leaks (T8).
- Fail any new `pytest.skip` without a tracked GitHub issue ID. Today the
  audit found 0 skip-abuse in the GPU bucket but 1 hard skip regression
  (`test_objectives_time` switched to `SIMSOPT_RUN_FIELD_TIMING=1`-gated).
- Run `pytest --collect-only` in CI and fail when a known-deleted upstream
  test (e.g. `test_arclength_variation_circle_planar`) re-appears under the
  same name with a different body.

## 6. What's genuinely solid (do not touch)

### Bucket 1 (M1 kernels)
- `tests/field/test_biotsavart_jax_parity.py` Taylor convergence helpers
  (`_assert_point_perturbation_taylor_convergence`,
  `_assert_second_derivative_taylor_convergence`),
  `test_B_is_curl_A`, `test_dB_dX_symmetric_and_divergence_free`,
  `test_B_linearity_in_current` (bit-exact), `test_B_parity_ncsx`,
  `test_dB_by_dX_parity_ncsx`.
- `tests/objectives/test_integral_bdotn_jax.py::test_zero_when_B_tangential` and
  `test_strict_oracle_scalar_reduction_matches_high_precision_reference`
  (`math.fsum` oracle).

### Bucket 2 (M3/M4 Boozer)
- `tests/geo/test_label_constraints_jax.py` — all 6 tests use real Taylor
  convergence with proper ε-ladder.
- `tests/geo/test_boozer_derivatives_jax.py::TestBoozerHessianComposed::test_hessian_taylor_convergence`
  (`min_observed_order=2.0`).
- `tests/geo/test_boozersurface_jax.py::TestVJPHooks::test_ls_reduced_directional_requires_spatial_field_derivatives`
  — proves the dB/dX term must be present via FD-vs-symmetric-FD with
  active-vs-dropped comparison.
- `TestVJPHooks::test_add_G_current_cotangent_matches_abs_vjp_at_zero_current`
  at `rtol=1e-14`.
- `TestBoozerExactConstraintsJacobianTaylor` (real exact residual, ratio=0.55
  ε-ladder over `2^-7..2^-19`).
- `TestUpstreamFactoryBoozerMatrix::test_penalty_gradient_taylor_matrix` (real
  NCSX surfaces, multi-epsilon Taylor).
- `TestStellsymMaskCPUJAXParity` (3 grid configs × 3 mpol/ntor/nfp combos).
- `TestNfpVolumeArea` (analytical `2π²Rr²` and `4π²Rr` at `rtol=1e-13`).

### Bucket 3 (M5 single-stage)
- `TestExactSolveCPUJAXParity::test_exact_coil_vjp_matches_fixed_state_directional_fd`
  (line 5679).
- `test_boozer_residual_wrapper_rejects_exact_surface` (line 5720).
- `TestEnsureSolvedCrashGuard::test_J_before_run_code_gives_clear_error`
  (line 5197), `test_m5_wrappers_raise_on_failed_solve_state` (line 5247).
- `TestCompositeGradientPipeline::test_branch_stable_composite_gradient_fd_and_descent`
  (line 4263) — gold-standard resolve-FD test; use as the template to harden
  the looser FD tests.
- `tests/integration/test_jax_native_path.py::TestGradientFiniteDifference`
  (lines 271, 314, 364) — three real FD validations on the SquaredFluxJAX
  path.

### Bucket 4 (Stage 2 / flux)
- `test_b_parity` (line 1574, `rtol=1e-10, atol=1e-15`),
  `test_dB_by_dX_parity` (line 1720), `test_B_and_dB_vjp_parity` (line 1752,
  uses `parity_ladder_tolerances("derivative_heavy")` SSOT),
  `test_b_pullback_native_projects_to_public_derivative` (line 1622,
  `rtol=1e-12, atol=1e-14`).
- `test_j_only_uses_forward_path_until_gradient_is_requested` (line 1105) —
  caching contract.
- `test_j_ignores_mutated_field_points_after_construction` (line 2121) —
  JIT-closure contract.
- `test_strict_mode_keeps_mixed_quadrature_squared_flux_on_native_lane`
  (line 2180) — strong negative coverage that monkey-patches `bs_jax.B` /
  `B_vjp` to raise.
- `test_fluxobjective_value_parity` / `test_fluxobjective_gradient_parity` /
  `test_non_rz_fixed_surface_value_and_gradient_parity` (CPU-vs-JAX at
  `1e-12 / 1e-11`).
- `test_squaredfluxjax_requires_native_field_contract` —
  `NotImplementedError` match on `coil_dof_extraction_spec`.

### Bucket 5 (backend / runtime / smoke)
- `tests/conftest.py::_guard_backend_runtime_state` — autouse fixture that
  snapshots all 28 backend env vars + `jax.config` and restores via
  `try/finally`.
- `test_audited_entrypoints_configure_runtime_before_importing_jax` (line 406)
  — AST-based ordering guard.
- `_assert_no_private_jax_src_usage` (line 1147) — AST-based forbidden-symbol
  guard.
- `test_maybe_initialize_distributed_jax_invalidates_preinit_chunk_caches`
  (line 1135) — proves cache invalidation actually moves the chunk policy.
- `test_single_stage_surface_reprojection_probe_emits_structured_cpu_result`
  (line 847) — gold-standard subprocess+JSON pattern.
- `_CompileCounter` / `_assert_run_solver_compiles_once`
  (`tests/subprocess/jax_runtime_cases.py:179-205`) — pins compilation count.

### Bucket 6 (GPU proof)
- Nothing in `test_hf_production_gpu_proof.py` exercises GPU code; the only
  artefacts to preserve are the launcher plumbing tests themselves once
  renamed to `test_hf_production_gpu_proof_shell.py`. The hand-built
  `tests/subprocess/section6_fixture_probe.py` schema and the
  argparse-rejection messages (lines 749, 871, 940, 970) are useful as
  launcher-CLI contract tests.

### Bucket 7 (core / curve / misc)
- `tests/geo/test_simsoptpp_compat.py:40-62`
  (`test_surface_xyztensorfourier_theta_third_derivative_matches_finite_difference`)
  — only solid numerical test in the bucket (tighten tolerance, do not delete).
- `tests/core/test_reductions.py::test_strict_oracle_scalar_mode_matches_high_precision_reference`
  — true `math.fsum` oracle that proves strict_oracle beats default.
- `tests/geo/test_accessibility.py:251-271`
  (`test_port_size_refreshes_cached_port_solve_on_parent_curve_mutation`) —
  genuine behavioural test of cache invalidation.
- `tests/geo/test_candidate_ledger.py:328-382` and
  `tests/geo/test_stage2_seed_report.py:141-163` — corrupt-JSON tolerance
  pinning.

### Bucket 8 (modified upstream)
- `tests/geo/test_curve.py::test_curvecwsfourier_matches_cpp_on_stage2_surface`
  (lines 1788-1792, gamma `atol=1e-14`, gammadash `1e-13`, kappa `1e-12`).
- `tests/geo/test_curve.py:1812-1813, 1827-1855` — `_lin` paired-method
  parity and 3rd-derivative FD.
- `tests/field/test_biotsavart.py:435-528` — 4 new fieldcache tests with
  `< 1e-15` residual.
- `tests/core/test_derivative.py:120-141` — convention-drift markers for
  JAX vs NumPy block hostification.
- `tests/geo/test_curve_objectives.py:248-261`
  (`test_pairwise_penalty_chunking_preserves_infeasible_barrier_inf`) —
  `+inf` propagation through chunking.

## 7. Open questions for the user

1. **Is the section-6 / HF production proof bundle schema frozen by an
   external consumer?** The P0 #1 fix extends the schema with
   `bundle_provenance`/`default_backend`/`devices`/`xla_flags`/parity fields.
   If a downstream Hugging Face dashboard or the Runpod sign-off pipeline
   already parses today's narrow schema (`{passed, elapsed_s, failures,
   missing_payload, corrupt_payload}`), the extension is non-trivial; we'd
   need a `schema_version` bump and a compatibility shim. If the schema is
   internal-only, the fix is straight-forward.
2. **Is the `_MockVolumeLabel.J() ≡ 0.0` behaviour deliberately silencing the
   constraint for a reason I'm missing (e.g., tests originally meant to
   exercise unconstrained Boozer)?** Or was it a placeholder that got reused?
   The fix in P0 #2 changes the gradient profile of ~70 % of M4 tests; some
   may need updated tolerances or new fixture state.
3. **Should `field.set_points()` after `SquaredFluxJAX` construction RAISE
   or be silently ignored?** CLAUDE.md says "INVALID"; the current contract
   (test at line 2121) is "silently ignored". A loud failure is friendlier
   to users but is a behaviour change. Pick one and write the test for it.
4. **Are the four pure-numerics optimizer tests
   (Rosenbrock at line 828, generic newton_polish_quadratic at line 970,
   newton_exact_linear_system at line 1124, traceable-operator-only at line
   1230 in `test_boozersurface_jax.py`) intended to live there as a
   convenience, or should they move to `test_optimizer_jax_generic.py`?**
   Moving them frees ~80 lines in the M4 file and clarifies that
   `TestOptimizerAdapter` is not a Boozer physics suite.
