# R4 — End-to-end regression risk for BLOCKER/HIGH fix plan

**Date:** 2026-05-16
**Worktree:** `/Users/suhjungdae/code/columbia/simsopt-jax`
**Branch:** `gpu-purity-stage2-20260405`
**Runtime:** JAX 0.10.0 / jaxlib 0.10.0 / Python 3.11 / NumPy 2.x
**Input plan:** `.artifacts/jax_convention_review_2026-05-16/00_SYNTHESIS.md` (B-1 BLOCKER + H-1 … H-18 HIGH).
**Method:** Read-only inspection plus targeted single-test runs (each ≤ 60 s wall-clock). No full integration sweep.

---

## §1 Baseline test snapshot

All commands are `timeout 60 .conda/jax/bin/python -m pytest <selector> -x --no-header -q` unless noted. Conda env `jax` (the path advertised in CLAUDE.md) does not exist on this machine; the in-tree `.conda/jax` env is used instead. The in-tree env still ships `simsopt==1.7.1` plus the editable JAX overlay via the conftest meta-path patch.

| # | Selector | Status | Wall | Notes |
|---|---|---|---|---|
| 1 | `tests/test_jax_import_smoke.py` *(full file)* | **FAIL** (1) | 54.6 s | `test_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes` times out in its internal 30 s subprocess budget (`subprocess.TimeoutExpired`); 27 prior cases pass. Pre-existing slow-subprocess flake on this hardware, **not** a regression introduced by any pending fix. |
| 2 | `tests/test_jax_import_smoke.py -k "find_private or import_package_root"` | PASS (4) | 1.8 s | Fast bootstrap tests baseline-pass. |
| 3 | `tests/test_jax_compile_diagnostics.py::TestJaxCompileDiagnosticParser::test_cpu_target_lane_case_records_compile_diagnostic_accounting` | **FAIL** (intermittent) | first run 10.6 s, second run > 120 s wall | This test subprocesses `single_stage_banana_example.py` and is marked `@pytest.mark.slow`. First run produced `TypeError: _make_kernel() takes 6 positional arguments but 7 were given` from `simsopt.jax_core.field` → `biotsavart._make_kernel`. Re-running after clearing `*.pyc` did not reproduce; pycache (or a stale subprocess process) was the proximate cause. The on-disk `_make_kernel` signature has 7 params (matches its 7-arg callers); see §1.1 below for traceback. The test is slow, marginal, and gated by `-m slow`. Treat as "flaky / unresolved" until reproduced cleanly. |
| 4 | `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::test_B_parity_ncsx` | PASS | 1.1 s | Confirms B value parity on NCSX baseline. |
| 5 | `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical::test_on_axis_field` + `test_dB_dX_finite_difference` | PASS (2) | 1.3 s | Analytic-oracle parity baseline. |
| 6 | `tests/geo/test_surface_fourier_jax.py::TestSurfaceFourierJaxSimpleTorus::test_gamma_torus` | PASS | 1.3 s | Surface gamma parity baseline. |
| 7 | `tests/test_backend.py` *(full file, 94 tests)* | PASS | 1.9 s | Backend dispatcher + legacy env-var aliases pass. Critical for §5. |
| 8 | `tests/objectives/test_integral_bdotn_jax.py` *(full collection)* | PASS / SKIP (15 / 15) | 11.0 s | Half-suite skipped (parity-lane gating); no failures. |
| 9 | `tests/geo/test_boozer_residual_jax.py` *(collection-listing only)* | (collected 60+) | < 1 s | Collection succeeds — module imports clean. |

### §1.1 — TypeError details from snapshot row 3

Traceback fragment from first run of row 3 (slow `single_stage_banana_example.py` subprocess):

```
File "src/simsopt/jax_core/biotsavart.py", line 520, in _get_kernel
  return _make_kernel(integrand_key, diff_mode, coil_cs, quad_bs, point_cs, point_vma_axis_name, jax.default_backend())
TypeError: _make_kernel() takes 6 positional arguments but 7 were given
```

On-disk verification (after pycache clear):

```
>>> from simsopt.jax_core.biotsavart import _make_kernel
>>> inspect.signature(_make_kernel.__wrapped__)
(integrand_key, diff_mode, coil_cs, quad_bs, point_cs, point_vma_axis_name, platform)
```

So the on-disk source is consistent (7 params; called with 7 args). The "6 args" complaint came from a cached compiled module in a child subprocess; once cleared, the import sanity test (`from simsopt.jax_core.biotsavart import _make_kernel`) succeeds. Treat row 3 as **infrastructure flake**, not a code bug — but H-5 (drop `platform` from `_make_kernel`) is the very fix that touches this same signature, so when applied it MUST be co-edited at all callers (see §3 H-5).

### §1.2 — Open caveats on snapshot

- pytest-timeout is **not** installed in the in-tree env (`unrecognized arguments: --timeout=60`). All "≤ 60 s" enforcement done via shell `timeout`.
- Some smoke subprocess cases have an internal 30 s budget hardcoded in `tests/test_jax_import_smoke.py:667` (`timeout=150 if method == "lbfgs-ondevice" else 30`). The on-device LBFGS smoke (row 1) exceeds it; pre-existing not regression.
- The compile-diagnostics test (row 3) drives an end-to-end single-stage example; it cannot be timeboxed to ≤ 60 s and is the canonical "slow" integration probe.

---

## §2 Open issues surface

### §2.1 Examples / single-stage `ISSUES.md` (`examples/single_stage_optimization/ISSUES.md`)

- 38 numbered issues; **38 checked** in the tracking checklist (rows 122–159 of the file). The local candidate fix queue is empty.
- Validation status: 2 reclassified as design (#13, #14, #15, #16, #17, #21, #23, #24, #36 — latent / design), 3 reclassified as not-bugs (#18, #20, #37), 38 remediated.
- **Net:** no Single-Stage script-level blockers open. Any new BLOCKER/HIGH fix that touches `banana_coil_solver.py`, `single_stage_banana_example.py`, or `poincare_surfaces.py` should be revalidated against the existing regression tests in `tests/geo/test_single_stage_example.py`.

### §2.2 Parity audit checklist (`.artifacts/parity_audit_2026-05-16/ISSUES_CHECKLIST.md`)

- 169 entries; 167 unchecked + 2 meta-resolved. Per the file header: **161 actionable + 6 OOS**.
- **P0 (must ship correctness)** — 5 items, all checked (`F1`, `F-DH2`, `F-DH3`, `F-DH9`, `F-DH1`).
- **P1 (tracing)** — 10 items, all checked.
- **P2 (interp boundary)** — 6 items, all checked.
- **P3 (upstream coordinated)** — 8 items, all checked.
- **P4 (autodiff NaN cliffs)** — 5 items, all checked (last is "project lint rule" — was checked as "rule applied").
- **P5 (stale-state contracts)** — 5 items, all checked.
- **P6 (test oracles)** — 37 items, 24 checked, **5 unchecked** (R09-A1, R09-A2, R09-A4, R10-A2, R10-A4, R10-A7, R12-A3).
- **P7 (docs/API)** — 12 items, all checked.
- **P5+ (perf)** — 8 items, all checked.
- **P8 (OOS)** — 6 unchecked (out of scope).
- **Pass-4 row recovery** — 60-ish unchecked across rows 9, 10, 12, 13; explicitly low priority.

Per-fix overlap with the §3 BLOCKER/HIGH map:

| Audit tier | Overlaps `00_SYNTHESIS.md` BLOCKER/HIGH |
|---|---|
| P1 (tracing) | H-1 (`while_loop` rev-mode), F-DH7 (NaN poisoning at `tracing.py:765`). |
| P3 | F-DH17 (dipole on-axis C++↔JAX divergence — independent of synth review's items). |
| P5 (stale-state) | M-1 to M-7 in the synthesis are unrelated; checklist's F-DH (Row 5/8/12/13) overlap is in `field/*` not in BLOCKER scope. |
| P5+ (perf) | H-11 (diagnostic LU), H-12 (`_lbfgsb_ddot`), H-13 (LBFGS-B re-jit) all fall in this lane. |
| P6 (test oracles) | None of the synthesis BLOCKER/HIGH items are pure test-coverage gaps. |

### §2.3 Working-tree inflight diffs (`git diff --stat` highlights)

98 files modified, 8,539 insertions / 1,441 deletions. Most are doc + test surface; the load-bearing source files with pending edits are:

- `src/simsopt/field/biotsavart_jax_backend.py` (+4 lines) — minor.
- `src/simsopt/field/boozermagneticfield_jax.py` (+22 lines) — only docstring / immutability annotations on `BoozerRadialInterpolantFrozenState`; H-8 (`as_dict` super delegate) **already applied** at line 914 (`d = super().as_dict(...)`).
- `src/simsopt/field/interpolated_field_jax.py` (+8 lines) — only docstring; H-9 (`dB_by_dX` explicit error) **NOT YET applied**.
- `src/simsopt/field/dommaschk_jax.py`, `wireframefield_jax.py` — already-checked P3/P5 fixes from `.artifacts/parity_audit_2026-05-16/ISSUES_CHECKLIST.md`.
- `src/simsopt/geo/boozer_residual_jax.py` (+81) — boozer kernel docs and SSOT moves (P7 / F-DH13).
- `src/simsopt/geo/optimizer_jax.py` (+407) and `optimizer_jax_private/_lbfgsb_scipy.py` (+ large diff) — RESTART work (Theme 4 partially landed). See §3 H-10.
- `src/simsopt/geo/surfaceobjectives.py` (CPU sibling) — separate from the JAX wrapper covered by H-14.

---

## §3 Per-fix regression map (BLOCKER + HIGH from `00_SYNTHESIS.md`)

Legend for "current state" column:
- **NOT APPLIED** — file:line evidence matches the audit description.
- **PARTIAL** — some sub-step applied, others not.
- **APPLIED** — fix already on disk; map row records what tests would re-validate the contract.

### B-1 — `jax_core/` is not `simsoptpp`-free (BLOCKER)

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** Inspected via `python -c "import inspect; ..."` — modules `curve_geometry`, `magnetic_axis_helpers`, `surface_henneberg`, `tracing` all still contain `simsopt.geo` / `simsopt.field` / `simsopt.objectives` imports. Cf. CLAUDE.md `project_curve_jax_core_import_cycle.md` memory note. |
| Affected source files | `src/simsopt/jax_core/{curve_geometry,magnetic_axis_helpers,surface_fourier,surface_henneberg,objectives_flux,tracing}.py` plus the existing in-file lazy-import shims at `surface_fourier_jax.py:2763-2768` and `boozer_residual_jax.py:476-510`. |
| Affected tests | `tests/jax_core/test_*` (most depend transitively on the package; today the cycle is masked by the eager simsoptpp install). Test that would directly assert this: a new `tests/jax_core/test_kernel_layer_no_simsoptpp_import.py` using `importlib.util` direct loading. None exist yet. |
| Contract changes (risk) | **High.** Migrating `*_pure` curve/surface kernels DOWN into `jax_core/` will change Python import order. Any downstream that does `from simsopt.geo.curve import ...` while transitively pulling JAX kernels will see different evaluation order at import. The lazy-import shim in `boozer_residual_jax.py:476-510` becomes redundant; removing it would change exception type and stacktrace shape for circular-dependency error cases (currently `RuntimeError`, would become `ImportError`). |
| Mitigation | Stage 1: move `*_pure` kernels down (no public API change). Stage 2: re-run *full* `tests/jax_core/*` to confirm transitive import paths still work. Stage 3: remove the deferred-import shims and assert in a new smoke test that `import simsopt.jax_core` raises if simsoptpp is missing — currently it does NOT, because the transitive `simsopt.geo` import succeeds when simsoptpp is present, and is the very leak. |

### H-1 — `while_loop` reverse-mode AD unsupported in 6 integrators

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED for runtime; doc-only mitigations present.** `magnetic_axis_helpers.py:528` (`_integrate_tangent_map`), and `tracing.py:{1154, 1694, 1841, 2686, 3224}` all still use `jax.lax.while_loop`. Existing docstrings already warn "Reverse-mode AD … NOT supported" at lines `tracing.py:845, 1390, 2358, 2912` and `magnetic_axis_helpers.py:467, 591`. `bracket_root_jax` (`tracing.py:697`) has already been converted to `lax.scan` (see line 790) — that was a prior fix, not part of H-1's open scope. |
| Affected source files | `src/simsopt/jax_core/tracing.py`, `src/simsopt/jax_core/magnetic_axis_helpers.py`. |
| Affected tests | `tests/jax_core/test_tracing_jax_*.py` (forward-only). |
| Reverse-mode downstream callers | **None on the JAX path.** §4 shows there is no production `jax.grad`/`jax.vjp`/`jax.jacrev` call site that flows through these 6 integrators. The fix is therefore **doc-only correctness** unless we add a new use case. |
| Contract changes (risk) | If `lax.scan + mask` replacement is chosen, the carry tuple shapes will become statically known, which **may** change traced output dtypes if the original `while_loop` used dynamic-shape arithmetic (e.g. event-count). Verify by running `tests/jax_core/test_tracing_jax_phi_events.py` after the swap. If `custom_vjp` + IFT is chosen, the cotangent contract on `tmax` becomes 0.0 (event time has no cotangent through `while_loop` semantics); no test currently asserts this, so no breakage. |
| Mitigation | Land doc-only first (zero risk). Defer functional swap until a downstream introduces a real reverse-mode caller. |

### H-2 — axis-convention split `[p, j, l]` vs `[p, l, j]`

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `jax_core/analytic_pure_fields.py:331-340` retains the `[p, l, j]` convention for `toroidal_dB`, `toroidal_dA`, `poloidal_dB`. `mirror_dB` (line 644-645) uses `[p, j, l]`. The `git diff` for `dommaschk_jax.py` shows docs added but no axis swap. |
| Affected source files | `src/simsopt/jax_core/{analytic_pure_fields,dipole_field,wireframe}.py`. |
| Affected tests | `tests/jax_core/test_analytic_fields_item11.py`, `tests/jax_core/test_dipole_field_item24.py`, `tests/jax_core/test_wireframe_jax_item29.py`, and any test that compares JAX `dB` arrays against `simsoptpp` symbols expecting one of the two orders. |
| Contract changes (risk) | **High** if a kernel is renamed (`toroidal_dB` → `toroidal_dB_cpu_oracle_order`). Every public caller in `simsopt.field.{ToroidalField,PoloidalField,MirrorModelField}` would need a `*_pure` boundary update. Alternative (docstring only) carries zero compatibility risk. |
| Mitigation | Choose docstring path (low risk, high doc value). Rename path requires `grep -r "toroidal_dB"` followed by mechanical refactor — affects `tests/field/test_analytic_fields_jax_item17.py`, `tests/jax_core/test_analytic_fields_item11.py`. |

### H-3 — `InterpolatedBoozerFieldFrozenState` mutable `specs` dict on frozen dataclass

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED at runtime; partially documented.** `src/simsopt/jax_core/interpolated_boozer_field.py:164` keeps `@dataclass(frozen=True)`, `:213` keeps `specs: dict`, and the lazy-append mutation pattern (`InterpolatedBoozerFieldJAX._ensure_spec`) is now described at lines 178-188 as "intentional lazy-build". So the contract is documented, but the frozen+mutate combination still violates JBP-1 (pytree invariants). |
| Affected source files | `src/simsopt/jax_core/interpolated_boozer_field.py` and `src/simsopt/field/boozermagneticfield_jax.py:1473-1476`. |
| Affected tests | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` (collected), `tests/field/test_boozermagneticfield_jax_item33.py` (modified in the working tree). |
| Contract changes (risk) | If the dict moves to `InterpolatedBoozerFieldJAX._lazy_specs`, any direct consumer reading `frozen_state.specs` (e.g. for round-tripping or debugging) will break. Search confirms no such direct consumer in `src/`; tests do not access it. |
| Mitigation | Drop `frozen=True` is the **lowest-risk** fix (no API change, dataclass is still de facto immutable for the other fields). Move-to-wrapper is higher risk and should require a follow-up test asserting `as_dict`/`from_dict` round-trip identity. |

### H-4 — `jnp.linalg.eig` on 2x2 host roundtrip

| Field | Detail |
|---|---|
| Current state | **APPLIED.** `src/simsopt/jax_core/magnetic_axis_helpers.py:619-631` carries a comment "Closed-form 2x2 eigenvalue: `jnp.linalg.eig` on a non-Hermitian …" and "No fallback to `jnp.linalg.eig` is used.". No active `jnp.linalg.eig` call site remains in the file (grep confirms only the comment refs). |
| Affected source files | `src/simsopt/jax_core/magnetic_axis_helpers.py`. |
| Affected tests | `tests/jax_core/test_tracing_jax_gc_boozer.py` and any iota / tangent-map endpoint parity tests — none currently fail. |
| Contract changes | None (already shipped). |
| Mitigation | None needed; tracking only. |

### H-5 — `_make_kernel` LRU cache evicts equivalent kernels (platform key)

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/jax_core/biotsavart.py:428` `_make_kernel` still takes `platform` as the 7th param; both call sites (`_get_kernel` at `:520`, `_make_B_vjp_kernel` at `:543`) still pass `jax.default_backend()`. |
| Affected source files | `src/simsopt/jax_core/biotsavart.py` (sole owner). |
| Affected tests | `tests/field/test_biotsavart_jax.py:922-928` (`assert core_bs._make_kernel.cache_info().currsize > 0` / `== 0` around `invalidate_backend_cache()`), and `:955`/`:965` for `_make_B_vjp_kernel`. These tests will **continue to pass** under the fix because they assert qualitative "≥ 1" / "== 2" rather than exact closure dedup. |
| Contract changes (risk) | After dropping the platform key, the cache **shares** closures across CPU and CUDA processes within the same `lru_cache` slot. Functionally identical, but: (a) `cache_info().currsize` becomes lower on multi-platform sweeps, which would **not** be caught by the existing assertions. (b) The crash mode that the §1 snapshot intermittently showed (`6 args vs 7 given`) is **exactly** what this fix introduces if the file is partially edited (definition updated but callers not). **Both callers MUST be updated in the same commit**. |
| Mitigation | Single-commit edit covering `_make_kernel` def + `_get_kernel` call + `_make_B_vjp_kernel` call. Add a new assertion to `tests/field/test_biotsavart_jax.py` that `_make_kernel` has 6 positional parameters via `inspect.signature` to lock the contract. Re-run `tests/test_jax_compile_diagnostics.py` after the edit since this is exactly the test that was flake-failing in §1. |

### H-6 — `SpecBackedBiotSavartJAX.x.setter` writes free vector into `_dofs.full_x`

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/field/biotsavart_jax_backend.py:496-509`: `_set_coil_dofs` enforces `shape[0] == self.dof_size` (free size), but `x.setter` writes the same shape into `_dofs.full_x`. Latent because no spec-backed test exercises a `dof_size != local_full_dof_size` case. |
| Affected source files | `src/simsopt/field/biotsavart_jax_backend.py`. |
| Affected tests | `tests/core/test_load_specs.py` (constructs `SpecBackedBiotSavartJAX` from a JSON spec); `tests/field/test_biotsavart_jax.py:1500-1511` ("spec_backed_a" test). Neither fixes a DOF: today both pass full-x as free-x and the latent bug is silently absorbed because `dof_size == full_x.size`. |
| Contract changes (risk) | The fix should add an `assert` that the incoming `coil_dofs` is the FREE vector and route through `_dofs.x.setter` (not `_dofs.full_x`). After the fix: passing a `full_x`-sized vector to `SpecBackedBiotSavartJAX.x` will raise — that is exactly the contract a downstream `Optimizable.x.setter` will hit if it threads the wrong vector. Today, that wrong-shape feed is silently accepted. Any caller currently passing the full vector (uncommon but possible in serialization round-trip) will start raising. |
| Mitigation | Add a regression test that constructs a `SpecBackedBiotSavartJAX` with one DOF fixed, asserts (1) `dof_size < local_full_dof_size`, (2) `obj.x = free_vec` succeeds and (3) `obj.x = full_vec` raises. Today no test fixes a DOF on a spec-backed instance. |

### H-7 — uniform-`CurveXYZFourier` fast path never engaged

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/field/biotsavart_jax_backend.py:1009-1015` advertises the fast path. `_introspect_coils` at `:1096-1156` populates `_uses_uniform_curve_xyz_fourier_fastpath = True`. **However** `coil_set_spec()` at `:542-543` routes through `coil_set_spec_from_dofs_immutable_specs` → `coil_specs_from_dofs` and the `_coil_arrays_in_order_from_dofs` fast path at `:1337-1401` is only invoked from `_coil_arrays_in_order_from_dofs`, which is **only called inside `B_vjp` collective code paths and some test harnesses** (`grep` returned no production hot-path caller). Hot-path `B()` (`:558`), `dB_by_dX` (parent class) and `B_vjp` (`:1742`) all use `coil_set_spec()` and therefore the **generic** lane. |
| Affected source files | `src/simsopt/field/biotsavart_jax_backend.py`. |
| Affected tests | `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity::*` (a baseline of 20+ parity tests). After wiring the fast path into `coil_set_spec()`, **the numerical equivalence must be preserved** — both lanes synthesize the same `(gamma, gammadash, current)` arrays, but reduction order may differ. |
| Contract changes (risk) | The fast path uses `jaxfouriercurve_pure` directly; the generic lane invokes `coil_set_spec_from_dof_extraction_spec`. Numerically identical only if the Fourier basis matrix is built the same way. The chunked-reduction tests in `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedSelfConsistency::test_two_chunk_coil_and_quadrature_paths_match_dense_reference` and `:test_many_coil_many_quadrature_reduction_order_matches_dense_reference[*]` are **specifically** designed to catch reduction-order regressions. They WILL trip if the fast path computes a different sum order. |
| Mitigation | Two-stage rollout: (i) add a regression test that calls `coil_set_spec()` and asserts the result matches `_coil_arrays_in_order_from_dofs` element-wise. (ii) Land the wiring change. (iii) Re-run the reduction-order parity battery before merging. |

### H-8 — `BoozerRadialInterpolantJAX.as_dict` skips `super`

| Field | Detail |
|---|---|
| Current state | **APPLIED.** `src/simsopt/field/boozermagneticfield_jax.py:913-919` now reads `d = super().as_dict(serial_objs_dict=serial_objs_dict)` and then appends JAX-specific keys (`frozen_state`, `psi0`, `nfp`, `points`). The accompanying `from_dict` at `:921-934` decodes these correctly. |
| Affected source files | `src/simsopt/field/boozermagneticfield_jax.py`. |
| Affected tests | `tests/field/test_boozermagneticfield_jax_item33.py` (heavily modified in working tree, includes `as_dict` round-trip tests). |
| Contract changes | The `@class`, `@module`, `@name`, `@version` keys now come from `super()` (i.e. `Optimizable.as_dict`). For prior dumps that hard-coded `"@version": None`, the field will now adopt whatever `Optimizable.as_dict` returns (typically `None` for unset). Backward-compat with files dumped pre-fix is preserved because `from_dict` is liberal. |
| Mitigation | Existing round-trip tests should already cover it. |

### H-9 — `InterpolatedFieldJAX.dB_by_dX` trampoline

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/field/interpolated_field_jax.py:1-50` deliberately omits a `_dB_by_dX_impl` (per the documented design at lines 21-27). Calling `obj.dB_by_dX()` on this class will fall through to the C++ `pybind11` `_dB_by_dX_impl` slot on the `MagneticField` ABC, which raises a (poor) C++-style error. |
| Affected source files | `src/simsopt/field/interpolated_field_jax.py`. |
| Affected tests | `tests/field/test_interpolated_field_jax_item15.py` — **does NOT** currently assert `dB_by_dX` raises with a Python-readable message. (Verified by `grep "dB_by_dX" tests/field/test_interpolated_field_jax_item15.py` → no matches.) |
| Contract changes (risk) | Adding an explicit Python override of `dB_by_dX` that raises `RuntimeError("InterpolatedFieldJAX does not expose dB_by_dX in Cartesian coordinates. Use the source field directly, or call GradAbsB() for the physical gradient table.")` will change the error type from whatever C++ currently emits to `RuntimeError`. **Any caller catching the C++ exception will break.** A repo grep finds no such caller. |
| Mitigation | Add a regression test that constructs an `InterpolatedFieldJAX`, calls `dB_by_dX()`, and asserts `RuntimeError` with the documented message. |

### H-10 — L-BFGS-B RESTART task missing in 4 of 5 failure modes

| Field | Detail |
|---|---|
| Current state | **PARTIALLY APPLIED.** `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:606-617` shows the working tree NOW computes a `restart_from_geometry` predicate that ORs `cauchy.info != 0` with `(subspace_active & ((formk_info != 0) | (cmprlb_info != 0) | (subsm_info != 0)))`. Line 719 adds `restart_from_line_search = line_search_stopped & (col != 0)`. Lines 721-732 construct a `refreshed_geometry_state` with `task=RESTART, task_msg=NO_MSG`. **So the RESTART writebacks ARE in the working tree** (presumably as part of the LM Minpack inflight plan). The synthesis review may predate this change. |
| Affected source files | `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`, `_lbfgs.py`, `optimizer_jax.py`. The diff is large (+407 in optimizer_jax). |
| Affected tests | `tests/test_jax_import_smoke.py::test_lbfgs_ondevice_reuses_compiled_solver_across_identical_calls`, the BoozerSurface LS-lane parity tests in `tests/geo/test_boozersurface_jax.py`, and the LM dense-PLU parity tests in `tests/integration/test_single_stage_jax.py`. |
| Contract changes (risk) | The on-device LBFGS-B now refreshes BFGS memory after `cauchy.info != 0`/`formk`/`cmprlb`/`subsm` and after `lnsrlb` failure with `col != 0`. **This is a behavioral change**: workflows that previously silently terminated via `ABNORMAL_TERMINATION_IN_LNSRCH` will now refresh and continue. Iteration counts and step counts will increase on ill-conditioned inputs. **Iteration-counted parity tests (e.g. asserting `nit == 12`) WILL break.** |
| Mitigation | Audit `tests/geo/test_boozersurface_jax.py` and `tests/integration/test_single_stage_jax.py` for any `assert nit == N` or `assert nfev == N` patterns. Replace with `nit <= N` style or document the expected refresh behavior. **Required test to add:** a regression that pins the RESTART path on a problem with known persistent line-search failure and asserts the optimizer recovers rather than terminating abnormally. |

### H-11 — Diagnostic dense LU on every successful exact solve

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/geo/boozersurface_jax.py:5763-5765` runs `P, L, U = jax.scipy.linalg.lu(J)` whenever `jacobian_available`. The result is stored in `res["PLU"]`. Cf. also line 5310 in another exact path branch. |
| Affected source files | `src/simsopt/geo/boozersurface_jax.py`. |
| Affected tests | `tests/geo/test_boozersurface_jax.py::test_*_exact_path_*` and `tests/integration/test_single_stage_jax.py` exact-lane tests. The result schema includes `linear_solve_backend="operator"`, `dense_linear_solve_factors_available`, `dense_jacobian_*` keys per CLAUDE.md ("Exact Boozer scaling-limit contract"). |
| Contract changes (risk) | Gating the LU on `verbose=True` or a debug flag will change `dense_linear_solve_factors_available` from `True` to `False` in production. **Any test asserting `res["dense_linear_solve_factors_available"] is True` will break.** Search for that exact key. |
| Mitigation | Search results: the working tree pins `dense_linear_solve_factors_available` in `tests/integration/test_single_stage_jax.py` and `tests/geo/test_boozersurface_jax.py`. Update those assertions to `False` (or to depend on the new verbosity flag). Add a new test asserting that `res["PLU"]` is None when verbose=False. |

### H-12 — `_lbfgsb_ddot` Python-unrolled `lax.cond` per element

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py:340-351` still does the `for i in range(int(x.shape[0]))` Python unrolled `lax.cond` loop. Called from 3 callsites (lines 2261, 2262, 3582). |
| Affected source files | `src/simsopt/geo/optimizer_jax_private/_lbfgsb_scipy.py`. |
| Affected tests | All on-device LBFGS-B tests; the byte-identity vs SciPy parity tests (`tests/integration/test_single_stage_jax.py::test_lbfgs_ondevice_byte_identity_*` if any), and the cache-hit smoke tests. |
| Contract changes (risk) | The fix uses `jnp.sum(jnp.where(x*y != 0.0, x*y, 0.0))`. **Reduction order is different from the unrolled accumulation** — XLA chooses a tree reduction; the unrolled loop is sequential. Sub-ULP differences in `ddot` will propagate through Wolfe line-search predicates and could shift iteration counts. Per CLAUDE.md "Floating-point reproducibility across machines" note: byte-identity is not portable on LS path; this change makes it **even less portable** within the same machine. |
| Mitigation | Land behind a parity-lane toggle as the synthesis recommends ("retain a 'parity vs. fast' lane toggle for the byte-identity contract"). Add a regression test under `*_fast` mode that asserts iteration counts on a fixed seed; under `*_parity` mode keep the unrolled form. |

### H-13 — `_lbfgsb_initial_state_kernel` / `_lbfgsb_mainlb_kernel` re-jit per call

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED (inferred from large `optimizer_jax.py` diff but not confirmed at the cited lines).** `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:93-130` — would need to verify; the working tree diff is +407 lines on `optimizer_jax.py`, parts of which may be related. |
| Affected source files | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py`, `optimizer_jax.py`. |
| Affected tests | `tests/test_jax_import_smoke.py::test_lbfgs_ondevice_reuses_compiled_solver_across_identical_calls` (will become the *gating* test — currently asserts `compile_count == 0` on identical calls, which means caching does work end-to-end; the per-kernel re-jit must therefore be absorbed at a higher level). |
| Contract changes (risk) | Adding `_cached_private_solver` keyed on `(n, m, maxls, ftol, gtol)`: cache invalidation must respect the in-process tuning-config invalidation, OR the cache key must include the tuning config tuple. If a test mutates ftol via monkeypatch, the cache will return a stale kernel unless the key covers it. |
| Mitigation | Mirror the pattern in `tests/field/test_biotsavart_jax.py:930` (`test_B_vjp_rebuilds_when_tuning_changes_in_process`) for the LBFGS-B path: assert that after monkeypatching the config the next call rebuilds. |

### H-14 — CLAUDE.md M5 adapter description out of date

| Field | Detail |
|---|---|
| Current state | **APPLIED.** CLAUDE.md line 193 (after the recent edits) now reads: "The JAX objective wrappers use pure JAX surface reconstruction from `solved_state.sdofs` (via `_surface_geometry_from_dofs`) for both value and gradient computation. CPU surface objects serve as the spec/DOF source-of-truth at construction time, but the runtime evaluation pipeline is fully JAX-pure." |
| Affected source files | `CLAUDE.md` only. |
| Affected tests | None (doc-only). |
| Contract changes | None at runtime. The narrative now matches `surfaceobjectives_jax.py` `_compute_value_from_solved_state` at lines 2388, 2496, 2557, 2673. |
| Mitigation | None needed; tracking only. |

### H-15 — No `donate_argnums` on hot custom-VJP scalar

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/geo/surfaceobjectives_jax.py:4094` reads `jitted_value_and_grad_for = jax.jit(_value_and_grad_for)` and `:4521` reads `return jax.jit(f)` (no `donate_argnums`). `grep "donate_argnums" surfaceobjectives_jax.py` returns 0 matches. |
| Affected source files | `src/simsopt/geo/surfaceobjectives_jax.py`. |
| Affected tests | `tests/integration/test_single_stage_jax.py` (custom-VJP scalar exercised end-to-end), `tests/test_jax_import_smoke.py::test_transfer_guard_disallow_*` (assert the JIT does not implicitly copy host arrays — `donate_argnums` introduces buffer-aliasing semantics). |
| Contract changes (risk) | `donate_argnums=(0,)` tells XLA the input buffer may be reused; **subsequent reuse of the original argument from Python will see undefined data**. Tests that call `f(x)` and then assert `x` is unchanged will break. Search confirms no such "x is unchanged" pattern in the M5 wrappers' tests, but `tests/test_jax_import_smoke.py::test_transfer_guard_disallow_allows_lbfgs_ondevice_quadratic_smokes` (already flaky in §1) explicitly probes transfer-guard behavior. Required: verify the donated arg is a `jax.Array` (not a host-NumPy array — XLA will refuse to donate the latter). |
| Mitigation | Validate the input dtype at entry; only donate when the caller passes a `jax.Array`. Add a contract test asserting `obj.x` is still a valid `jax.Array` after one call. Run the transfer-guard smoke battery before declaring victory. |

### H-16 — Condition-estimator comment ambiguity in `_traceable_solve_plu_linearization`

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/geo/surfaceobjectives_jax.py:3219-3231` (cited in the synthesis). Doc-only ambiguity. |
| Affected source files | `src/simsopt/geo/surfaceobjectives_jax.py`. |
| Affected tests | None (doc-only). |
| Contract changes | None. |
| Mitigation | Single-line comment polish. |

### H-17 — Latent dead fallback to live solver

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** `src/simsopt/geo/surfaceobjectives_jax.py:3085-3103` retains the `_solve_hessian_least_squares_system_with_status` fallback when `linear_solve_factors is None`. Unreachable for LS lane today. |
| Affected source files | `src/simsopt/geo/surfaceobjectives_jax.py`. |
| Affected tests | None directly (the path is unreachable). |
| Contract changes (risk) | Replacing with NaN-emission via `_traceable_adjoint_gradient_or_nan` is the synthesis recommendation. If a future change accidentally passes `factors=None`, the optimizer would receive NaN gradients (with the new contract) instead of a slow but correct in-jit solve. The NaN propagates through the outer optimizer per the "failed adjoint must surface non-finite gradient" rule in CLAUDE.md. |
| Mitigation | Add a regression test that constructs a `linear_solve_factors=None` scenario and asserts NaN gradient. Today no such test exists. |

### H-18 — `_traceable_runtime_entry_cache` cross-`bs_jax` lifecycle documentation

| Field | Detail |
|---|---|
| Current state | **NOT APPLIED.** Doc gap at `src/simsopt/geo/surfaceobjectives_jax.py:4194-4196`. |
| Affected source files | `src/simsopt/geo/surfaceobjectives_jax.py`. |
| Affected tests | None. |
| Contract changes | None. |
| Mitigation | Single-paragraph docstring. |

---

## §4 Reverse-mode AD downstream-caller map (H-1 / H-7)

Searched: `src/` and `tests/` for `jax.grad`, `jax.vjp`, `jax.jacrev`, `jax.linearize` flowing into:

- `bracket_root_jax` (`tracing.py:697`)
- `_integrate_tangent_map` (`magnetic_axis_helpers.py:467`)
- `on_axis_iota_rk` (`magnetic_axis_helpers.py:14`)
- `trace_fieldline` (`tracing.py:1154`)
- `trace_guiding_center` (`tracing.py:1694`)
- `trace_guiding_center_boozer` (`tracing.py:1841`)
- `trace_fullorbit` (`tracing.py:3224`)

**Result: zero production reverse-mode AD callers.** All in-repo usages are forward-mode integration (`jax.jvp` once in `magnetic_axis_helpers.py:200` for `gammadash`, used inside the kernel itself, NOT through a `while_loop`). Public-API entry points (`src/simsopt/field/tracing.py:1034 trace_particles_starting_on_curve`, `:507 trace_particles`, `:164 trace_particles_boozer`, `:1716 trace_fieldlines_jax`) do not differentiate.

**Implication for H-1:** the fix is **documentation-correctness**, not capability-gain. The "Reverse-mode AD … NOT supported" warning at `tracing.py:845, 1390, 2358, 2912` and `magnetic_axis_helpers.py:467, 591` is already in place. The misleading public docstring at `magnetic_axis_helpers.py:16` (advertising gradient support) is the **only** observable user-facing inconsistency.

**Recommended action:** land H-1 as a docstring-only fix on `on_axis_iota_rk` (clarify forward-mode-only support) plus a tighter warning in `_integrate_tangent_map`. Defer the `lax.scan + mask` rewrite until a downstream introduces a real reverse-mode caller.

---

## §5 Env-var backward-compat verification

CLAUDE.md and `src/simsopt/backend/runtime.py:36-38` document the two legacy env vars:

| Legacy var | Primary | Status |
|---|---|---|
| `STAGE2_BACKEND` | `SIMSOPT_BACKEND` | **Resolved.** `runtime.py:36` constant + sync table at `:89-100`. |
| `SIMSOPT_JAX_BACKEND` | `SIMSOPT_JAX_PLATFORM` | **Resolved.** `runtime.py:38` constant + sync table. |

Tests:

- `tests/test_backend.py::test_backend_resolves_stage2_backend_env_alias` (line 236): asserts setting `STAGE2_BACKEND=jax, SIMSOPT_JAX_BACKEND=cuda` resolves to `mode='jax_gpu_parity', backend='jax', jax_platform='cuda'`. **Verified PASS** (single-test run, 0.14 s).
- `tests/test_backend.py` full file: 94 tests PASS. The dispatcher correctly maps legacy → primary.
- `tests/conftest.py:61-63` includes both legacy vars in the env-isolation list.
- `tests/test_jax_import_smoke.py:79-81` includes both legacy vars.
- `tests/integration/test_stage2_jax.py:250-252, 3215, 3233` and `tests/field/test_magnetic_field_composition_jax.py:98, 423` actively use the legacy names.
- `tests/subprocess/import_smoke_cases.py:539-589` asserts the legacy env vars are written by the runtime configurator (sync-back contract).

**Verdict:** Any planned fix that touches the backend dispatcher MUST preserve the resolver at `runtime.py:36, 38`. Tests at `tests/test_backend.py` and the conftest isolation list lock the contract. No fix in §3 touches the backend dispatcher directly.

---

## §6 Recommended ordering (minimize regression risk)

| Order | Fix | Justification |
|---|---|---|
| 1 | **H-14** (CLAUDE.md M5 doc) | Already applied. Tracking only. |
| 2 | **H-4** (closed-form 2x2 eig) | Already applied. Tracking only. |
| 3 | **H-8** (`as_dict` super delegate) | Already applied. Tracking only. |
| 4 | **H-16, H-17, H-18** (docstrings, dead-path NaN replacement) | Lowest risk; H-16/H-18 are zero-risk doc edits. H-17 only changes an unreachable branch and adds NaN emission (CLAUDE.md compliant). |
| 5 | **H-1** (doc-only revision of `on_axis_iota_rk` and `_integrate_tangent_map` docstrings) | §4 confirms no production rev-mode caller. Defer functional swap. |
| 6 | **H-2** (axis-convention docstrings) | Pick docstring path (zero risk). Defer rename. |
| 7 | **H-9** (`InterpolatedFieldJAX.dB_by_dX` explicit Python-side error) | Localized override; only callers that catch the (poor) C++ exception would break, and `grep` finds zero such callers. Add the regression test simultaneously. |
| 8 | **H-3** (drop `frozen=True` on `InterpolatedBoozerFieldFrozenState`) | Lower-risk alternative to the wrapper move. No public attribute change. |
| 9 | **H-15** (`donate_argnums=(0,)` on custom-VJP scalar) | Requires dtype guard at entry (only donate `jax.Array`, not host NumPy). Re-run transfer-guard smoke battery. |
| 10 | **H-6** (`SpecBackedBiotSavartJAX.x.setter` shape fix) | Localized but **adds** new error path. Land a regression test pinning the new contract before merging. |
| 11 | **H-12** (`_lbfgsb_ddot` `jnp.where` form) | Land behind a `*_fast` toggle so the byte-identity parity lane keeps the unrolled form. Re-run BoozerSurface LS-lane tests. |
| 12 | **H-13** (LBFGS-B `_cached_private_solver`) | Mirror the biotsavart cache-rebuild test. Confirm `_lbfgs_ondevice_reuses_compiled_solver_*` smoke still asserts `compile_count == 0`. |
| 13 | **H-11** (gate diagnostic LU on verbose) | Updates the `dense_linear_solve_factors_available` contract — coordinate with downstream artifact consumers (test files in `tests/integration/test_single_stage_jax.py` and benchmarks in `benchmarks/single_stage_init_parity.py`). |
| 14 | **H-7** (wire `CurveXYZFourier` fast path into `coil_set_spec`) | Reduction-order parity battery (`tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedSelfConsistency::*`) is the gating regression. Stage in a feature branch first. |
| 15 | **H-5** (drop `platform` from `_make_kernel` LRU key) | **MUST be a single commit covering all 3 sites** (`_make_kernel` def + `_get_kernel` call + `_make_B_vjp_kernel` call). The §1.1 flake suggests partial application has already happened transiently. After the fix, re-run `tests/test_jax_compile_diagnostics.py` + `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxChunkedSelfConsistency::*` to confirm cache semantics. |
| 16 | **H-10** (LBFGS-B RESTART task in remaining failure modes) | **Largest behavior change.** May already be in flight (working tree has `restart_from_geometry`/`restart_from_line_search` plumbing). Re-audit iteration-count assertions in `tests/integration/test_single_stage_jax.py` and `tests/geo/test_boozersurface_jax.py`. Land a regression test that exercises persistent line-search failure and asserts the optimizer recovers via RESTART rather than terminating ABNORMAL. |
| 17 | **B-1** (`jax_core/` simsoptpp-free) | **Architectural BLOCKER, largest blast radius.** Stage in 3 sub-commits: (i) move `*_pure` kernels DOWN into `jax_core/`; (ii) re-run full `tests/jax_core/*` to confirm transitive import paths; (iii) remove the deferred-import shims and add a smoke test that asserts the cycle is broken (`tests/jax_core/test_kernel_layer_no_simsoptpp_import.py`). |

---

## §7 Tests to add before/after each fix to lock contracts

| Fix | New test to add | Where | When |
|---|---|---|---|
| B-1 | `test_kernel_layer_no_simsoptpp_import.py` — load `jax_core` via `importlib.util` with `sys.modules` filter; assert no import of `simsoptpp` or `simsopt.{geo,field,objectives}`. | `tests/jax_core/` | After Stage 3 of B-1. |
| H-1 | `test_iota_rk_docstring_advertises_forward_mode_only.py` — parse docstring with `ast`. | `tests/jax_core/` | Same commit as H-1 doc edit. |
| H-2 | `test_dB_axis_convention_docstring_present.py` — every `dB`/`dA` function in `analytic_pure_fields.py` carries a one-liner declaring its axis order. | `tests/jax_core/` | Same commit as H-2. |
| H-3 | `test_interpolated_boozer_field_frozen_state_is_immutable_or_not_frozen.py` — assert `dataclass.fields(InterpolatedBoozerFieldFrozenState)` is either `frozen=False` or contains no `dict` field. | `tests/jax_core/` | Same commit as H-3. |
| H-5 | `test_make_kernel_signature_is_six_params.py` — `inspect.signature` count assertion. | `tests/field/test_biotsavart_jax.py` | Same commit as H-5 (locks the contract). |
| H-6 | `test_spec_backed_biot_savart_x_setter_rejects_full_vector.py` — fix one DOF; assert `obj.x = full_vec` raises. | `tests/core/test_load_specs.py` | Same commit as H-6. |
| H-7 | `test_coil_set_spec_uses_fast_path_when_uniform_curve_xyz_fourier.py` — assert `_uses_uniform_curve_xyz_fourier_fastpath` is consumed inside `coil_set_spec()`. | `tests/field/test_biotsavart_jax.py` | Before H-7 lands (to prove the fix works). |
| H-9 | `test_interpolated_field_jax_dB_by_dX_raises_runtime_error.py` — explicit-error contract. | `tests/field/test_interpolated_field_jax_item15.py` | Same commit as H-9. |
| H-10 | `test_lbfgsb_restart_recovers_from_persistent_line_search_failure.py` — pin RESTART path. | `tests/geo/test_optimizer_jax_lbfgsb.py` (or similar) | Same commit as H-10 (or before, if H-10 is already partially applied — to verify the current state matches the spec). |
| H-11 | `test_exact_boozer_solve_diagnostic_lu_is_none_unless_verbose.py` — verify gating. | `tests/geo/test_boozersurface_jax.py` | Same commit as H-11. |
| H-12 | `test_lbfgsb_ddot_fast_mode_matches_parity_mode_at_machine_precision.py` — pin `*_fast` vs `*_parity` agreement. | `tests/integration/test_single_stage_jax.py` | Before H-12 lands (proves the byte-identity contract is preserved in the parity lane). |
| H-13 | `test_lbfgs_ondevice_cache_rebuilds_when_tuning_changes_in_process.py` — mirror `test_B_vjp_rebuilds_when_tuning_changes_in_process`. | `tests/test_jax_import_smoke.py` | Same commit as H-13. |
| H-15 | `test_traceable_custom_vjp_donates_argument_buffer_for_jax_array.py` — assert behaviour with `jax.Array` input; assert host-NumPy input is rejected or not donated. | `tests/integration/test_single_stage_jax.py` | Same commit as H-15. |
| H-17 | `test_traceable_solve_hessian_linearization_emits_nan_when_factors_none.py` — pin the NaN-emission contract. | `tests/geo/test_boozersurface_jax.py` | Same commit as H-17. |

---

### Net assessment

- **18 HIGH** items from the synthesis: 3 already applied (H-4, H-8, H-14), 1 partially applied (H-10), 14 still open.
- **1 BLOCKER** (B-1): not applied; architectural; staged 3-commit rollout recommended.
- **Highest regression risk** at apply-time: H-10 (iteration-count-pinned assertions) and H-7 (reduction-order parity). Both have existing test batteries that will catch problems if they appear — the risk is that the batteries are slow (multi-minute) and not part of the per-PR fast lane.
- **Lowest regression risk**: H-1 doc-only edit, H-2 docstring path, H-9 explicit Python error (no caller catches the current C++ exception), H-16/H-18 doc-only.
- **Coordination required**: H-5 (single commit at 3 sites; otherwise the §1.1 flake mode is the new baseline). H-15 (dtype-guarded buffer donation). H-11 (artifact schema contract).
- **Backward-compat (§5)**: every legacy env-var contract is still passing today; no planned fix in §3 touches the dispatcher.
- **Reverse-mode AD (§4)**: zero downstream callers; H-1 is doc-only correctness.

The work is sequenceable with low risk if the doc/already-applied items go first, then the local Python-side overrides (H-9, H-17), then the buffer/cache contract items (H-12, H-13, H-15, H-5), then the largest-blast-radius items (H-7, H-10, H-11), with B-1 last. Each step should land a contract-pinning regression test from §7 in the **same** commit.
