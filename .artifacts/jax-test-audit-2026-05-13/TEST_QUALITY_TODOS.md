# JAX Test Quality Remediation — TODOs

**Audit date:** 2026-05-13 (revised after independent validation)
**Audit scope:** JAX-ported tests under `tests/` on branch `gpu-purity-stage2-20260405`, HEAD `cadc6139e`.
**Findings:** 25 total (was 21; +4 from validation pass, #20 folded into #7), grouped by severity. All findings verified against source at audit time.
**Note:** Findings #1–#4, #17–#18, #21 reference files that exist only in the worktree (untracked). The cheapest fix window is **before** these files get committed. Finding #19 was incorrectly marked untracked in the prior revision — it is tracked.

## Revision log
- 2026-05-13 v2: validated and corrected. Added #22-#25 from independent review. Fixed #19 status (tracked, not untracked). Corrected #7 site list (removed L674 helper, added L1510 case). Folded #20 into #7.
- 2026-05-13 v3: corrected two remediation defects flagged by second-pass review:
  - **#24** plan pointed at the unconstrained minimum (`x = 1`, infeasible) and incorrectly claimed the multiplier should be ≈ 0. The constraint is `x ≤ 0` so the constrained optimum is `x = 0` with an *active* multiplier `μ ≈ 2`. Plan now anchors to the constrained KKT optimum.
  - **#7** grep-based acceptance gates were broken (one-line pipeline returns no matches despite 16 silent sites; the optional pre-commit grep overmatches legitimate branch returns). Replaced with explicit AST audit per AI-1 and a note that grep is not a reliable gate.
- 2026-05-13 v4: corrected one further #24 defect:
  - **#24** v3 plan referenced `result.total`, which does not exist on the `minimize_alm` result object. The real field names (verified at `examples/single_stage_optimization/alm_utils.py:1473, 2143`) are `result.final_objective`, `result.final_base_objective`, `result.final_penalty_objective`, and `result.history[*]["objective_delta"]`. Plan now lists the actual attributes and a `evaluate_problem(...)["total"]` recompute as alternatives.

---

## How to use this document

1. Work top-to-bottom within each tier — Tier 1 first.
2. Each finding has a **Context** (what's wrong and why), a **Plan** (concrete remediation steps as checkboxes), and a **Done-when** acceptance criterion.
3. Mark `[x]` as steps complete. When a finding's done-when is met, mark `[x]` on the finding header.
4. Where a fix replaces a tautological test with a real one, the goal is **delete or rewrite**, not "add another test". The codebase already has too many tests; the issue is signal-per-test, not test count.

---

## Suggested execution order

1. **Block the propagation** (#7 silent subprocess skips, #21 export trivia) — these patterns are still spreading. Fix the infrastructure first or new cases will keep landing.
2. **Tier 1 tautologies** (#1-#6) — these claim coverage they don't have. Each is in a closeout artifact that downstream agents trust.
3. **Tier 2 hacky tests** (#8-#13) — labels/markers are wrong.
4. **Tier 3 weak tests** (#14-#19) — loosen-as-needed honesty.
5. **Tier 5 self-reporting / mislabeled tests** (#22-#25) — added in v2 revision after independent validation pass.

---

# Tier 1 — Tautological tests (always pass by construction)

## [ ] #1 — `tests/geo/test_framedcurve_jax_item18.py:182-227` — same function compared to itself

**Status:** untracked (`mtime` 2026-05-13)

**Context:**
`src/simsopt/geo/framedcurve.py:10-19` directly re-exports `rotated_centroid_frame` and `rotated_frenet_frame` from `simsopt.jax_core.framedcurve`. The test imports both names and asserts they produce the same output — but they are the *same Python function object*. `upstream_rotated_centroid_frame is rotated_centroid_frame` evaluates to `True`. Two parametrized tests (`test_rotated_centroid_frame_matches_upstream`, `test_rotated_frenet_frame_matches_upstream`) are pure tautology. The docstring claim "pin the new kernels to bit-identity with the upstream JAX evaluation" is false because there is no upstream.

The orthonormality and α=0 reduction tests (L230-333) in the same file are real and should be kept.

**Plan:**
- [ ] Delete `test_rotated_centroid_frame_matches_upstream` (L182-202).
- [ ] Delete `test_rotated_frenet_frame_matches_upstream` (L205-227).
- [ ] Replace with a C++/CPU-oracle parity test if item 18 needs a parity row. The CPU `FramedCurveCentroid`/`FramedCurveFrenet` classes route through the same JAX kernels, so a real oracle needs to be either (a) a closed-form analytic frame (planar circle has known centroid frame), or (b) a snapshot from `simsoptpp` for `FrameRotation`-touching kernels.
- [ ] Update the file docstring to drop the "bit-identity with upstream JAX evaluation" claim.

**Done-when:** No test in this file compares a function to itself. Item 18 closeout coverage rows are anchored to an independent reference.

---

## [ ] #2 — `tests/geo/test_framedcurve_jax_wrappers_item18.py:70-197` — JAX-vs-JAX disguised as host parity

**Status:** untracked (`mtime` 2026-05-13)

**Context:**
`test_frame_rotation_jax_matches_host`, `test_zero_rotation_jax_matches_host`, `test_framed_curve_frenet_jax_matches_host`, `test_framed_curve_centroid_jax_matches_host` — the "host" `FramedCurveFrenet`/`FramedCurveCentroid` invoke the same JAX kernels the JAX wrappers do (via the same re-exports flagged in #1). So `_jax_matches_host` is `JAX_wrapper(x) == host_wrapper(x)` where both call the same JAX kernel under the hood.

The DOF round-trip test (L91-99) and dependency-graph test (L200-212) are real (they test Optimizable contract, not kernel correctness) — keep them.

**Plan:**
- [ ] Delete the four `*_jax_matches_host` tests.
- [ ] Add a real Optimizable-contract test: verify `wrapper.J()` and `wrapper.dJ()` change correctly when DOFs change (FD gradient check, not function-vs-self).
- [ ] If kernel correctness for `FrameRotation`-side wrappers is needed, anchor against analytic planar-circle frames or a numpy reference.

**Done-when:** No `*_matches_host` test in this file.

---

## [ ] #3 — `tests/geo/test_finitebuild_jax_ssot_item20.py:56-146` — same JAX kernel, different staging

**Status:** untracked (`mtime` 2026-05-13)

**Context:**
`CurveFilament.gamma()` already runs through JAX (`finitebuild.py:60-66` jits `gamma_jax`). The SSOT helper `build_filament_gammas` then routes through the same JAX path. Tests `test_build_filament_gammas_matches_create_multifilament_grid` and `test_build_filament_gamma_and_dash_matches_grid_first_filament` are JAX-vs-JAX with no independent oracle.

The pure offset-calculation test `test_compute_filament_offsets_matches_grid_construction` (L149-173) is fine (it tests Python arithmetic, not the JAX hot path).

**Plan:**
- [ ] Delete the two `build_filament_gammas_matches_*` tests.
- [ ] If item 20 needs parity-row coverage, add a closed-form analytic test for filament offsets on a simple geometry (e.g., known centroid frame on a planar circle with hand-computed offset positions).

**Done-when:** Only one test on `build_filament_gammas`, and it has an independent oracle.

---

## [ ] #4 — `tests/geo/test_finitebuild_jax_item20.py:132-182` — JAX VJPs vs JAX derivatives

**Status:** untracked (`mtime` 2026-05-13)

**Context:**
`test_curvefilament_jax_vjps_match_public_derivative_methods` compares two JAX-backed VJP routes (Optimizable derivative vs spec-pullback). Both are JAX. No independent gradient oracle.

**Plan:**
- [ ] Replace JAX-vs-JAX comparison with finite-difference vs JAX (the existing FD-validated pattern in `test_boozer_derivatives_jax.py` is the right template).
- [ ] Use a small DOF count so FD is tractable; an FD-vs-JAX agreement at `rtol=1e-6` is a real gradient check.

**Done-when:** This test asserts JAX gradient against FD, not against another JAX route.

---

## [ ] #5 — `tests/test_single_stage_cpp_jax_state_parity.py:78-112` — GPU lane is a copy-pasted CPU lane

**Status:** tracked, commit `bee115caed` (2026-05-04)

**Context:**
`_cpu_artifact_with_fake_cuda_lane()` literally does:
```python
cuda_artifact["lanes"][LANE_JAX_GPU] = dict(cuda_artifact["lanes"][LANE_JAX_CPU])
```
Then `test_fixed_state_merge_builds_complete_fixed_state_artifact` asserts `comparisons["cpp_cpu_vs_jax_gpu"]["status"] == "pass"`. The "GPU parity" assertion is structurally guaranteed because the GPU lane is the CPU lane relabeled. This file tests **artifact-merge logic**, not GPU parity. The filename is misleading and downstream artifacts cite it as evidence of CPU/GPU parity.

**Plan:**
- [ ] Rename file to `test_state_artifact_merge_logic.py` (or move tests into an existing merge-logic test).
- [ ] Update the docstring at file head to say "tests merge logic; real CPU/GPU JAX parity is in `test_single_stage_jax_cpu_reference.py::TestRealFixtureGpuM5Parity`".
- [ ] Search for citations of this file in audit reports (`.artifacts/jax_port_goal/` and bench JSONs) and correct them.
- [ ] Add a real CPU/GPU JAX state parity test using subprocess fan-out (CPU side runs on host JAX, GPU side runs on actual CUDA via `subprocess.run` with `SIMSOPT_JAX_PLATFORM=cuda`).

**Done-when:** No test file in `tests/` claims CPU/GPU parity while constructing the GPU side by relabeling CPU output.

---

## [ ] #6 — `tests/geo/test_strainopt_item08_closeout.py:65-138` — host oracle reimplements the formula

**Status:** tracked, commit `1b57237fc3` (2026-05-12)

**Context:**
`_numpy_lp_torsion_reference` docstring says "Host-side NumPy reproduction of `Lp_torsion_pure`". Same formula, NumPy vs JAX, on the same `frame_torsion()` / `gammadash()` arrays. A wrong formula in `Lp_torsion_pure` (or `torstrain_pure`) would pass the test silently. The test catches only JIT/FP-order drift.

**Plan:**
- [ ] Replace `_numpy_lp_torsion_reference` with a closed-form test fixture where the integrand is hand-computable. For example: a torsion-free planar curve has `Lp_torsion_pure = 0`; a curve with known constant torsion has `Lp_torsion_pure = (1/p) * (|τ| - threshold)^p * total_arc_length` when `|τ| > threshold`.
- [ ] If no closed form is tractable, anchor against `simsoptpp` strain CPP routines (whatever the upstream non-JAX path uses); the import already exists in the surrounding test suite.
- [ ] Drop the redundant `assert ... or ...` line at L130-132 (the following `assert_allclose` is strictly tighter).

**Done-when:** The oracle is independent of `Lp_torsion_pure`'s implementation (closed form or `simsoptpp`).

---

# Tier 2 — Hacky tests (hidden skips, snapshot-as-truth, mislabeled gates)

## [ ] #7 — `tests/subprocess/jax_runtime_cases.py` — 16 silent-skip sites, no JSON sentinel

**Status:** tracked across multiple commits (2026-04-12 → 2026-05-13). **Growing**: was 11 sites at last audit, now 16.

**Context:**
Every `if not _configure_strict_cpu_parity_backend(): return` and every `if gpu is None: return` silently exits the subprocess with returncode 0. The pytest harness reports PASSED because it only checks the subprocess returncode, not whether assertions ran. On CPU-only hosts, ~9 subprocess-driven test cases report PASSED with zero assertions. The audit script `tests/test_pytest_skip_xfail_audit.py` cannot see across the subprocess boundary because it's an AST scanner.

**Current sites (16 total):** L237, L290, L346, L561, L747, **L1510**, L1523, L1550, L1641, L2089, L2100, L2117, L2140, L2164, L2188, L2211. The previously listed L674 is the `_configure_strict_gpu_fast_backend()` helper returning a sentinel `None` — that is correct API and is **not** a silent-skip site; callers see the sentinel and currently do a silent `return`. L1510 (inside `_run_gamma_2d_eager_host_constants_case`) is the silent-skip case my prior revision missed.

**Plan:**
- [ ] Add a `_skip_case(reason: str)` helper modeled on `tests/subprocess/import_smoke_cases.py:32-33`. It should emit a JSON sentinel: `{"case": ..., "skipped": true, "skip_reason": <str>, "checked": false}`.
- [ ] Replace every `if not _configure_strict_cpu_parity_backend(): return` with `if not _configure_strict_cpu_parity_backend(): _skip_case("strict CPU parity backend unavailable"); return`. Same for `if gpu is None: ...`.
- [ ] Update `_assert_python_script_passes` in `tests/test_jax_import_smoke.py` (or wherever the harness is) to parse the sentinel and call `pytest.skip(reason)` upstream.
- [ ] Update `tests/test_pytest_skip_xfail_audit.py` to extend coverage: walk `tests/subprocess/*.py` and require either `_skip_case(...)` with a non-empty reason or no early-`return` from a top-level `_run_*_case` function.
- [ ] Add an **AST-based** CI gate (see AI-1) that walks `tests/subprocess/*.py`, finds every top-level `def _run_*_case(...)` function, and fails if any `If` node inside it has a body whose final statement is a bare `Return(value=None)` that is not immediately preceded by a `_skip_case(...)` call.
  - **Do not use a one-line grep gate.** The v2 revision proposed `grep -nE "if (not .*\(\)|gpu is None):\s*$" tests/subprocess/ | grep -B1 -A1 "return$"` — that pipeline returns zero matches today despite the 16 silent sites still existing, because the first grep only emits the if-lines and the second grep's `-B1 -A1` runs on stdin (the if-lines), not on the original file. The gate is silently broken.
  - The AST audit is the only reliable acceptance check for this pattern.

**Done-when:** Zero silent `return` from `_run_*_case` functions; every skip surfaces as `pytest.skip()` in the test report; the audit script catches new silent-skip attempts.

---

## [ ] #8 — `tests/geo/test_curve_item05_closeout.py:220-225` — pytest.skip after real assertions

**Status:** tracked, commit `ffada5daba` (2026-05-12)

**Context:**
The test asserts `gamma.shape`, `np.isfinite(gamma).all()`, and `pytest.raises(NotImplementedError)` — then unconditionally calls `pytest.skip(...)`. Pytest reports SKIPPED, masking that real regression coverage exists. If the source class later grows `to_spec()`, the `pytest.raises` would fail — but the report would still show SKIPPED.

**Plan:**
- [ ] Decide: is "blocker still in place" worth a passing test or an `xfail(strict=True)`?
  - If passing test: delete the trailing `pytest.skip(...)` (L220-225). Rename the test to something like `test_curvexyzfouriersymmetries_does_not_expose_immutable_spec`.
  - If xfail: convert to `@pytest.mark.xfail(strict=True, reason="...")` plus a positive assertion that `curve_spec_from_curve(curve)` succeeds. The xfail strict mode will fail the test if the blocker disappears, which is the right signal.
- [ ] Update the file docstring to match the chosen pattern.

**Done-when:** Pytest reports PASSED (or XFAIL with strict) for this test, not SKIPPED.

---

## [ ] #9 — `tests/geo/test_boozer_residual_pinned_input_byte_parity.py:366-417` — fail-by-design without xfail

**Status:** tracked, commit `9460c81cf7` (2026-05-09); fixture rewritten in unstaged diff (improves test surface but issue stands).

**Context:**
Four tests assert byte-identity between CPU and JAX residual value/gradient and full-penalty value/gradient. The file docstring at L18-26 documents that they are expected to FAIL today until P4.3 closes an FMA gap. The docstring at L354 explicitly refuses xfail: "do not commit xfail as the red test". Sibling "drift within current baseline" tests (L429-478) accept up to `5e-13` gradient drift. So neither byte parity nor a real numerical tolerance is enforced — only a "current drift level" is pinned.

**Plan:**
- [ ] **Decide the gate**: is byte-identity the actual ship gate, or is `5e-13` the actual ship gate?
  - If byte-identity: fix the FMA arrangement (P4.3 work in `boozer_residual_jax.py`) so the four assertions pass.
  - If `5e-13`: delete the four byte-parity tests; rename the drift tests from `*_drift_within_current_baseline` to `*_within_drift_ceiling` with the ceiling values explicitly chosen and documented as the ship gate.
- [ ] If the project intentionally wants "audit-visible failing test", use `@pytest.mark.xfail(strict=True, raises=AssertionError)` and update `tests/test_pytest_skip_xfail_audit.py` to accept this specific test as an allow-listed strict-xfail entry. Strict xfail is **not** the same as `skip` — it asserts the failure shape.

**Done-when:** Either CI is green with byte parity, or the ship gate is a documented drift ceiling and the byte tests are gone. No tests fail by design without an explicit `xfail(strict=True)` marker.

---

## [ ] #10 — `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py:251-265` — upstream failure converted to skip

**Status:** tracked, commit `e50bc4cedb` (2026-05-12)

**Context:**
When upstream `simsopt.load()` returns `verdict == "unsupported"` for `CurveCWSFourier`, the test calls `pytest.skip(...)`. The skip reason says "Re-enable once the upstream JSON deserializer learns the CurveCWSFourier schema". This silently masks any future change in the JAX side's "unsupported" detection.

**Plan:**
- [ ] Convert the second `pytest.skip(...)` (L260-265) to `pytest.xfail(strict=True, reason="...")`. If the deserializer fix lands, the xfail will fail the test, which is the right signal to re-enable.
- [ ] Keep the first `pytest.skip` at L252 (artifact-files-missing skip is principled).

**Done-when:** The test surface flips from SKIPPED to XFAIL today, and to PASSED when the upstream is fixed.

---

## [ ] #11 — `tests/test_jax_import_smoke.py:169-187` — JSON-sentinel translator's downstream cases

**Status:** tracked, commit `8f637eceac` (2026-04-26)

**Context:**
The `_assert_subprocess_json_sentinel` helper is *correct* — it translates a subprocess-emitted skip into `pytest.skip(reason)`. But ~50+ tests use this helper, and each downstream subprocess case can independently decide to emit a skip. Audit doesn't enumerate which cases skip on which conditions, so the surface is opaque.

**Plan:**
- [ ] Enumerate skip-emitting cases. Grep `tests/subprocess/import_smoke_cases.py` for `_skip_case(...)` and document each case's skip preconditions in `.artifacts/jax-test-audit-2026-05-13/SUBPROCESS_SKIP_INVENTORY.md`.
- [ ] For any case that skips on conditions other than "JAX unsupported runtime" or "GPU absent on CPU host", convert to a hard failure (a skip mask in a parity-critical case is a hidden regression).
- [ ] After #7 lands, this audit covers both `import_smoke_cases.py` and `jax_runtime_cases.py`.

**Done-when:** A single-file inventory documents every skip path across subprocess cases with its precondition. Pytest output reasons match the inventory.

---

## [ ] #12 — `tests/field/test_biotsavart_jax_cpu_ordered.py:150-202` — no-regression-as-parity

**Status:** tracked, commit `e61370cdf7` (2026-05-08)

**Context:**
`test_field_terms_parity_policy_routes_through_cpu_ordered` asserts only `cpu_ordered_drift ≤ production_drift + 1e-18`. It does not assert an absolute byte-identity bound on `cpu_ordered`. If both production and cpu_ordered drifted by `1e-3` from C++, the test would still pass. Hard parity is enforced in adjacent tests in the same file (L81-117) at `< 5e-14`, but the routing test in isolation gives misleading coverage.

**Plan:**
- [ ] Add a second assertion: `np.max(np.abs(B_cpu - B_cpp)) < <absolute ULP ceiling>` and `np.max(np.abs(dB_cpu - dB_cpp)) < <absolute ULP ceiling>`. Use the same ceiling as the sibling "within ULP of cpp" test.
- [ ] Keep the no-regression check too (it catches a different bug class — cpu_ordered regressing vs production).
- [ ] Rename the test to `test_field_terms_parity_policy_routes_through_cpu_ordered_and_meets_ulp_ceiling`.

**Done-when:** The routing test enforces both regression-freedom and absolute parity to C++.

---

## [ ] #13 — `tests/geo/test_surface_fourier_jax_cpu_ordered.py:279-324` — same pattern

**Status:** tracked, commit `e61370cdf7` (2026-05-08, same commit as #12)

**Context:** Same as #12 — `test_parity_policy_routes_through_cpu_ordered_kernels` asserts only `cpu_drift ≤ prod_drift + 1e-18`. It even has a comment saying "if it doesn't, the test fixture is too small to surface the bug" — i.e., the test deliberately picks fixtures where production drifts from C++.

**Plan:**
- [ ] Same fix as #12: add absolute ULP-ceiling assertion on `cpu_ordered_drift`. Reference the sibling `test_surface_gamma_cpu_ordered_matches_cpp_within_ulp` for the right ceiling.
- [ ] Reconsider whether the "production must diverge" assertion at L318-321 (`assert np.max(np.abs(prod - cpp_gamma)) > 0.0`) is needed; that's an assertion the system under test is broken, which is fragile.

**Done-when:** Both regression and absolute ceiling are enforced; no test asserts that the production path must be broken.

---

# Tier 3 — Weak or misleading tests

## [ ] #14 — `tests/integration/test_single_stage_physics_parity.py:667-803` — ceilings only on geometry metrics

**Status:** tracked, commit `cc38e4a442` (2026-04-21) + `03a3243c76` (2026-05-12)

**Context:**
`_assert_outer_loop_single_step_consistency` cross-checks `final_objective`, `final_volume`, `final_iota` between CPU and JAX lanes, but `mean_abs_bdotn_over_b`, `curve_curve_distance`, `curve_surface_distance`, `banana_curve_max_curvature` are only checked against absolute physical ceilings, not cross-lane. Two divergent trajectories that both stay under ceilings would pass.

**Plan:**
- [ ] Decide for each metric: is it inherently lane-dependent (different optimizers), or should it converge across lanes?
  - `mean_abs_bdotn_over_b` is a quality metric on the same field — it should be cross-lane checkable.
  - Distances/curvatures are hard constraints — ceilings are the right check.
- [ ] Add cross-lane assertion for `mean_abs_bdotn_over_b` at a loose-but-honest rtol (e.g., `5e-3`).
- [ ] Document in the helper docstring exactly which metrics are cross-lane and which are ceilings-only, and why.

**Done-when:** Every metric in the helper has documented cross-lane status; quality metrics are cross-checked.

---

## [ ] #15 — `tests/geo/test_strainopt_item08_closeout.py:143-170` — "vanishes identically" with `1e-10` floor

**Status:** tracked, commit `1b57237fc3` (2026-05-12)

**Context:**
Test asserts `value <= max(_ATOL, _RTOL) = 1e-10` for a "zero-twist circle" case where the docstring says strain "vanishes identically". A buggy kernel returning `5e-11` from real (non-vanishing) arithmetic passes silently. The circle radius is `1e-4`, so the natural scale is small, but a true zero should be machine-zero.

**Plan:**
- [ ] Tighten the assertion to `value == 0.0` if the math truly vanishes (zero-twist Frenet circle → binormal curvature is zero by construction).
- [ ] If FP rounding makes a hard zero impossible, document the smallest expected residual from theory and use that as the ceiling. `1e-10` is not derived from anything.

**Done-when:** The tolerance is either `0.0` (exact) or anchored to a specific FP-error analysis.

---

## [ ] #16 — `tests/integration/test_stage2_jax.py:188, 1339, 1340` — three "parity" tests at `rtol=1e-2` to `1e-3`

**Status:** tracked, commits `2e37083fe8` (2026-03-28), `1bc05140f7` (2026-04-02)

**Context:**
- `_SHORT_RUN_PARITY_RTOL = 1e-3` (`test_short_run_parity`)
- `_PHYSICS_PARITY_RTOL = 1e-3` (`test_physics_quantities_at_convergence`)
- `_BASIN_OBJ_PARITY_RTOL = 1e-2` (`test_basin_stability`)

These are smoke tests labeled "parity". The strict same-state direct-kernel parity (`rtol=1e-10..1e-12`) is enforced elsewhere; these "parity" labels are misleading.

**Plan:**
- [ ] Rename the three tests to drop "parity" and add "smoke" or "convergence":
  - `test_short_run_parity` → `test_short_run_convergence_smoke`
  - `test_physics_quantities_at_convergence` → keep name but rename the constant from `_PHYSICS_PARITY_RTOL` to `_PHYSICS_CONVERGENCE_RTOL`
  - `test_basin_stability` → keep name; rename `_BASIN_OBJ_PARITY_RTOL` to `_BASIN_OBJ_CONVERGENCE_RTOL`
- [ ] Update docstrings to clarify that these are convergence/smoke tests, not byte-identity gates.

**Done-when:** No test labeled "parity" enforces `rtol > 1e-6`.

---

## [ ] #17 — `tests/jax_core/test_tracing_jax_item14.py:69-123, 474-494` — discarded analytic anchor + JIT smoke

**Status:** untracked (`mtime` 2026-05-13)

**Context:**
- `test_dopri5_step_recovers_analytic_solution_against_scipy`: computes `np.exp(0.1)` but the gate is JAX-vs-SciPy with SciPy at `rtol=1e-3`. Comment at L116 says the analytic value is "severity context (not the gate)". The real anchor is computed and discarded.
- `test_trace_fieldline_jit_compiles_and_runs`: only asserts `isfinite(endpoint)` and `steps_taken > 0`. No value oracle.

**Plan:**
- [ ] In `test_dopri5_step_*`: assert JAX result against `np.exp(0.1)` directly at a tighter tolerance (single-step truncation error of dopri5 is known a-priori — pick the analytic ceiling).
- [ ] In `test_trace_fieldline_jit_compiles_and_runs`: rename to `test_trace_fieldline_jit_runs_without_error` (honest name for what it does); OR add a value oracle (e.g., field-line on a known axisymmetric field has known closed-form geometry).

**Done-when:** Either the analytic anchor is the gate, or the test honestly states "JIT smoke only".

---

## [ ] #18 — `tests/field/test_boozermagneticfield_jax_item33.py:121-128, 203-215` — trivia

**Status:** untracked (`mtime` 2026-05-13)

**Context:**
- `test_freeze_boozer_radial_state_returns_pytree`: asserts leaves are `jax.Array` — structural trivia.
- `test_wrapper_is_not_exported_via_field_namespace_lazy_init`: negative export check.
- `test_wrapper_has_no_dofs`: `size==0` check.

Each costs nothing to run but adds noise. They pass for any non-broken module.

**Plan:**
- [ ] Decide whether these structural checks belong in this file or in a single `test_*_exports.py` (see #21). If kept, leave a comment explaining what *behavior* is being protected (e.g., "if `field.BoozerMagneticFieldJAX` becomes eagerly imported, the JAX backend leaks into CPU-only environments").
- [ ] Delete the ones that don't protect a documented behavior.

**Done-when:** Every remaining structural test has a comment naming the regression it would catch.

---

## [ ] #19 — `tests/test_lightning_production_gpu_proof.py:373-376` — silent skip on missing SDK

**Status:** tracked, commit `7b428851a8` (2026-05-12). *(prior revision incorrectly noted "untracked"; corrected here)*

**Context:**
`test_terminal_failure_statuses_match_lightning_sdk_enum` tries to import `lightning_sdk` and skips if absent. The test exists specifically to catch drift between hardcoded status strings and the SDK enum — but if CI runs without the SDK, the drift goes undetected.

**Plan:**
- [ ] Replace the silent skip with an env-var-gated hard requirement:
  ```python
  if not os.environ.get("LIGHTNING_SDK_AVAILABLE"):
      pytest.skip("set LIGHTNING_SDK_AVAILABLE=1 in lightning-capable CI")
  from lightning_sdk import Status  # ImportError here is a CI bug
  ```
- [ ] In CI config (`.github/workflows/`), set `LIGHTNING_SDK_AVAILABLE=1` for the lightning-relevant jobs only.

**Done-when:** Missing SDK in lightning-CI is a hard failure, not a silent skip.

---

# Tier 4 — Meaningless trivia tests (new findings)

## [merged into #7] #20 — silent-skip propagation

This entry was folded into #7 during the 2026-05-13 v2 revision. The "pattern is still spreading" observation belongs in the #7 plan as a propagation note, not as a separate finding. **No action here — see #7.**

Optional add-on for #7 (carried over from old #20):
- [ ] Add a pre-commit hook that blocks new silent `return` statements in `tests/subprocess/*.py`.
  - **Do not use a grep-based hook.** The v2 revision proposed `grep -nE "^\s+return\s*$" tests/subprocess/*.py | grep -vE "_skip_case|return  # explicit"`. That overmatches legitimate branch returns (e.g., `def _find_gpu_device(): ...; return None` style helpers, early-returns after successful processing). The grep would produce false positives on every legitimate `if not ready(): return` in production code and on every helper that returns `None`.
  - The right gate is the AST audit described in AI-1. The hook should call into the same Python-based audit walker the CI runs, so a single source of truth defines what "silent skip" means.

---

## [ ] #21 — `tests/solve/test_jax_solve_exports.py` + `tests/geo/test_permanent_magnet_grid_jax_exports.py` — `module.foo is other_module.foo`

**Status:** both untracked (`mtime` 2026-05-13)

**Context:**
Two new files, 23 + 16 lines total, contain only re-export identity checks like:
```python
def test_permanent_magnet_jax_solve_helpers_are_public_exports():
    assert solve.relax_and_split_jax is pm_jax.relax_and_split_jax
    ...
```
These verify Python's `from X import Y` mechanism. They pass for any non-broken import. Zero correctness coverage.

**Plan:**
- [ ] Decide what regression these protect against. If the goal is "ensure `simsopt.solve` exposes these names publicly", that is already enforced at the package level — the import succeeds or fails at test collection time. The `is`-identity check only catches a developer who explicitly aliases instead of re-exporting, which is a non-issue.
- [ ] **Recommended: delete both files before committing.**
- [ ] If the export contract really needs a test, write **one** parametrized test in an existing location (e.g., `tests/test_jax_import_smoke.py`) that enumerates the public-API surface against a known list of names.

**Done-when:** Neither file exists; if export coverage is needed, it lives in a single enumerated-API test.

---

# Tier 5 — Self-reporting and bookkeeping tests (added in v2 revision)

## [ ] #22 — `tests/integration/test_single_stage_physics_parity.py:852` — CUDA proof asserts probe's own verdict

**Status:** tracked

**Context:**
`TestSingleStageOuterLoopGpuProof.test_cuda_outer_loop_probe_converges_under_strict_transfer_guard` (L852) builds `payload = _run_single_stage_outer_loop_probe(...)` and then asserts:
```python
assert payload["passed"] is True
assert payload["failures"] == []
```
plus a series of provenance/probe field equality checks. The `payload["passed"]` and `payload["failures"]` fields are produced by the same probe driver that fills in the other payload values — so the test asserts the probe's own self-reported verdict, with no independent recompute or oracle. The test does check `objective_decreased is True`, `objective_decrease > 0.0`, and `self_intersecting is False`, which are real (probe-internal) facts, but the headline "passed" assertion is circular.

**Plan:**
- [ ] Replace `assert payload["passed"] is True` with the specific physics-content assertions the probe is supposed to encode (objective decrease ratio, final-state finiteness, transfer-guard violations count == 0). Move the "passed/failures" verdict out of the test or assert *the components* that build the verdict, not the verdict field itself.
- [ ] Pin `probe["objective_decrease"]` (or the ratio `final/initial`) against a recorded baseline so a no-progress run fails the test loudly.
- [ ] Add an independent re-check: after the probe completes, re-evaluate the objective on the returned final DOFs (separate code path) and compare to `probe["final_objective"]`. This is the missing oracle.

**Done-when:** No assertion is `payload["passed"] is True`. Final-state physics is asserted via a re-computation path independent of the probe driver.

---

## [ ] #23 — `tests/integration/test_single_stage_physics_parity.py:891` — compile-diagnostic bookkeeping mislabeled as physics parity

**Status:** tracked

**Context:**
`TestSingleStageOuterLoopCompileSmoke.test_cpu_target_lane_case_records_compile_diagnostic_accounting` lives in `test_single_stage_physics_parity.py` but asserts:
```python
assert isinstance(diagnostics, dict)
assert compile_event_count >= 0
assert sum(compile_targets.values()) == compile_event_count - compile_target_parse_miss_count
```
These are bookkeeping/parsing-consistency checks on JAX compile diagnostics. They prove the diagnostic recorder is internally consistent, not that any physics is correct. A reader scanning `test_single_stage_physics_parity.py` for parity evidence will count this test toward parity coverage when it offers none.

**Plan:**
- [ ] Move this test out of `test_single_stage_physics_parity.py` into a dedicated `tests/test_jax_compile_diagnostics.py` (or merge into `tests/test_benchmark_helpers.py` if it already exists there).
- [ ] Rename the class from `TestSingleStageOuterLoopCompileSmoke` to `TestJaxCompileDiagnosticParser` so the intent is "this verifies the diagnostic recorder/parser invariants" rather than "single-stage outer-loop coverage".
- [ ] Update any closeout artifacts citing this test as physics-parity evidence (likely in `.artifacts/jax_port_goal/`); it is **not** that.

**Done-when:** No bookkeeping/diagnostic test lives in a file named `*_physics_parity.py`; closeout artifacts cite it correctly (as instrumentation evidence, not physics).

---

## [ ] #24 — `tests/geo/test_single_stage_alm_integration.py:633` — ALM progress gate accepts barely-any progress

**Status:** tracked

**Context (corrected in v3):**
The ALM test starts at `x = [1.5]` and asserts `result.x[0] < 1.5`. That accepts any downward motion, even one line-search step from 1.5 → 1.4999. The remaining assertions (multiplier and constraint-value finiteness) say nothing about optimization quality.

**Problem structure (verified against source at L593-633):**
- Objective: `base_total = (x - 1)^2` — unconstrained minimum at `x = 1`.
- Constraint: `constraint_value = x`, `feasibility = max(x, 0)`. Feasible region is `x ≤ 0`.
- Constrained optimum: minimize `(x-1)^2` subject to `x ≤ 0` → optimum is on the boundary at **`x = 0`** (NOT `x = 1` — that's infeasible).
- The constraint is **active** at the optimum. From KKT: `∇L = 2(x-1) + μ = 0` at `x = 0` → **`μ = 2`** (active multiplier, not zero).

The v2 plan said "constraint should be inactive, assert `abs(x - 1.0) < 1e-3`, multiplier ≈ 0". That was wrong — it pointed at the infeasible unconstrained minimum and incorrectly characterized the multiplier.

**Plan:**
- [ ] Assert convergence to the constrained optimum: `abs(result.x[0]) <= settings.feasibility_tol` (or a small multiple of it). At `x = 0` the constraint is just-feasible.
- [ ] Assert feasibility: `result.constraint_values[0] <= settings.feasibility_tol` (the feasibility-violation channel from `evaluate_problem` returns `max(x, 0)`).
- [ ] Assert multiplier is active and positive: `result.multipliers[0] > 0`. The KKT-consistent value is `≈ 2`; pick a loose check like `0.5 < result.multipliers[0] < 5.0` to allow for finite-iteration drift, or assert `abs(result.multipliers[0] - 2.0) <= <tol>` if the test budget is tight enough for KKT convergence.
- [ ] Assert objective decrease is meaningful: `(1.5 - 1.0)^2 = 0.25` is the starting *primal* objective and `(0 - 1.0)^2 = 1.0` is the constrained-optimum primal objective — so the **primal** objective actually *increases* as x → 0 (because the unconstrained minimum is infeasible). Assert the **augmented-Lagrangian total** decreased instead, using one of:
  - `result.final_objective` (defined at `examples/single_stage_optimization/alm_utils.py:1473` as `float(evaluation["total"])` — i.e., the Lagrangian total at the final outer iteration). Compare to a baseline-recompute of `evaluate_problem(np.asarray([1.5]), initial_multipliers, initial_penalty)["total"]` to assert a real decrease.
  - `result.history[*]["objective_delta"]` (defined at `alm_utils.py:2143` as `current_total - final_total` per outer iteration). Assert `sum(h["objective_delta"] for h in result.history) > <threshold>` for a meaningful net Lagrangian decrease.
  - An explicit recompute: `evaluate_problem(result.x, result.multipliers, result.penalty)["total"]` after the run, compared to the same call evaluated at the starting point. This is the most defensible because it does not depend on internal field naming.
  - There is **no** `result.total` attribute — do not assert against it.
  - As a coarse fallback if the test must stay simple, replace `assertLess(result.x[0], 1.5)` with a margin assertion like `result.x[0] < 0.1` so a trivial single line-search step does not pass.
- [ ] Keep the multiplier/constraint-value finiteness checks.

**Done-when:** The progress assertion is anchored to the *constrained* optimum (`x ≈ 0`, `μ ≈ 2`), not the starting point minus epsilon and not the infeasible unconstrained minimum.

---

## [ ] #25 — `tests/integration/test_single_stage_jax_cpu_reference.py:2388+` — bucket of health-only / routing-only tests

**Status:** tracked

**Context:**
Multiple tests in this file are health-only or routing-only rather than parity:

- **L2388 `TestBoozerResidualValue::test_j_both_small`** — asserts `j_jax < 1e-3` and `j_cpu < 1e-3` (ceilings, no cross-lane comparison). Two divergent residuals could both pass.
- **L2415 `test_value_path_matches_residual_helper_not_penalty_objective`** — monkeypatches `_value_and_direct_coil_gradient`, `_solve_boozer_adjoint`, `_adjoint_coil_dofs_gradient` to raise, then asserts the wrapper's value matches `_boozer_residual_J_of_x_inner(...)` *called from the test with the same args*. This is a routing test (proves the wrapper takes the residual-helper path), not a value-correctness test, because the "expected" is produced by the same helper the SUT is supposed to call.
- **L2484 `test_direct_objective_value_and_grad_is_cached_per_instance`** — counts factory invocations (`build_count == 1` after multiple `.J()`/`recompute_bell()` calls). Cache-count internal test.
- **L2519 `test_constraint_weight_is_concrete_float_for_ls_surface`** — `isinstance(..., float)` check. Structural trivia.
- **L2540 `TestIotasValue::test_j_finite`** — only asserts `np.isfinite(j_cpu) and np.isfinite(j_jax)`. Docstring explicitly says "Branch-divergent small-grid coverage stays health-only" — i.e., it knows it isn't doing parity.
- **L2585 / L2604 `test_device_native_adjoint_solve_satisfies_runtime_operator` / `test_adjoint_residual`** — both compute `adj = solve_transpose(dJ_ds)` and then assert `apply_transpose(adj) - dJ_ds ≈ 0`. This proves the operator is self-consistent (forward × inverse ≈ identity on dJ_ds), which is trivially true for any consistent linear operator and its inverse — it is *not* parity with a dense reference solve.
- **L2630 `test_vjp_produces_finite_derivative`** — `isfinite(g) and ||g|| > 0`. Health-only.

**Plan:**
- [ ] For each test, decide its honest classification: parity, smoke, routing, or instrumentation.
- [ ] **`test_j_both_small`**: either add cross-lane assertion (`relative_error(j_jax, j_cpu) < <rtol>` for the branch-stable lane) or rename to `test_j_both_below_health_ceiling` and reference it from a docstring saying "ceiling-only coverage; cross-lane parity is in <X>".
- [ ] **`test_value_path_matches_residual_helper_*`**: keep but rename to `test_value_routes_through_residual_helper_not_penalty_objective`. The "matches" claim is misleading — it routes, it does not match an independent value.
- [ ] **`test_direct_objective_value_and_grad_is_cached_per_instance`**: keep — caching is a real perf invariant. Rename `TestBoozerResidualValue` enclosing class to split out caching-vs-value subclasses, since this test is about caching, not value.
- [ ] **`test_constraint_weight_is_concrete_float_for_ls_surface`**: delete unless this catches a known historical bug (boundary float vs jax.Array). If kept, add a docstring naming the bug class it protects against.
- [ ] **`test_j_finite`**: rename to `test_j_finite_branch_divergent_smoke` and reference the branch-stable parity lane.
- [ ] **`test_device_native_adjoint_solve_satisfies_runtime_operator` / `test_adjoint_residual`**: keep — adjoint residual self-consistency is the documented Tier-4 gate. Rename file-level class docstring from "validate the exposed adjoint linear system" to "validate adjoint residual self-consistency (Tier-4 gate, not vector parity)". Move "vector parity" to the `exact-well-conditioned-adjoint` lane test if it doesn't already exist there.
- [ ] **`test_vjp_produces_finite_derivative`**: rename to `test_vjp_runs_and_produces_finite_nonzero_derivative`. Mark as smoke.

**Done-when:** Every test in this bucket has an honest name and a docstring naming the gate-tier (parity / smoke / routing / Tier-4 self-consistency). Downstream closeout artifacts cite each correctly.

---

# Audit infrastructure tasks

## [ ] AI-1 — Strengthen the AST audit to catch silent subprocess skips

**Context:**
`tests/test_pytest_skip_xfail_audit.py` is excellent at what it does — it audits visible `pytest.skip/xfail` calls. But it cannot see across subprocess boundaries, and findings #7/#20 are entirely in subprocesses.

**Plan:**
- [ ] Extend the audit walker to load each `tests/subprocess/*_cases.py` and audit top-level `_run_*_case` functions for bare `return` statements not preceded by a sentinel emission.
- [ ] Add a `_skip_case` import requirement: any case file that contains `_run_*_case` functions must `from .skip_sentinel import _skip_case` and use it consistently.

**Done-when:** Audit covers subprocess case files; CI fails on new silent returns.

---

## [ ] AI-2 — Make closeout artifacts cite real evidence

**Context:**
Findings #1–#4 are in closeout artifacts (item 18, 20). The downstream artifact files in `.artifacts/jax_port_goal/` cite these tests as evidence the items are done. Tautological tests passed → item marked done → false confidence propagates.

**Plan:**
- [ ] Audit `.artifacts/jax_port_goal/plans/18.md` and `20.md` (and the SUMMARY) to identify which "covered by" claims point to the four tautological files.
- [ ] After fixing #1-#4, re-run the closeout evidence pass and update the artifacts.
- [ ] Add a closeout-template requirement: every parity-row claim must cite (a) the test file, (b) the oracle (C++ symbol, closed-form expression, or external dataset), and (c) the parity-ladder tolerance lane used.

**Done-when:** Every parity-row claim in JAX port closeouts cites an independent oracle.

---

## [ ] AI-3 — Add a "what's the oracle" lint to the test reviewer flow

**Context:**
Every Tier-1 finding could have been caught by asking "what's the independent oracle here?" The pattern is repeatable.

**Plan:**
- [ ] Add a section to the `code-review` / `crucible` skill prompts: "For every assertion of equality or near-equality, name the independent oracle. If the oracle is a re-implementation of the system under test, flag the test as tautological."
- [ ] When new `test_*_jax_*.py` files are added, the reviewer runs this lint before approving.

**Done-when:** New JAX tests get the oracle question asked during review.

---

# Summary checklist (high level)

- [ ] Tier 1 (#1-#6) — six tautological tests deleted or rewritten with independent oracles
- [ ] Tier 2 (#7-#13) — silent-skip sentinel infrastructure; misleading "parity" labels corrected; xfail-strict where appropriate
- [ ] Tier 3 (#14-#19) — honest names and tolerances; cross-lane vs ceiling distinctions documented
- [ ] Tier 4 (#21) — export-identity trivia deleted (#20 folded into #7 during v2 revision)
- [ ] Tier 5 (#22-#25) — circular self-verdict assertions replaced with independent oracles; bookkeeping moved out of physics-parity files; ALM progress gate anchored; routing/health/cache tests renamed
- [ ] AI-1, AI-2, AI-3 — audit infrastructure prevents these patterns from re-landing

When the high-level checklist is done, re-run this audit and produce `TEST_QUALITY_TODOS_2026-XX-XX.md` to confirm no new patterns emerged.
