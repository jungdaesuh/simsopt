# BoozerSurface LS Port — Issue Resolution Plan

| Field | Value |
|---|---|
| Created | 2026-05-15 |
| Branch | `gpu-purity-stage2-20260405` |
| Tree at first probe | `cab64a15` (initial empirical measurements) |
| Tree at third review | `7822c5e0302cde8cbff8a7bd04b2143161051ae1` (current HEAD; reviewer's rerun) |
| Driver context | Deepdive on JAX BoozerSurface LS path; three independent review passes |
| Status | Waves 1-3 implemented in `10e92ef24` / `db23b9acf`; Wave 4 remains conditional |

Note: the checkbox lists below are the original execution checklist retained
for provenance. The issue-inventory status table is the current completion
summary.

---

## TL;DR

The CPU↔JAX disagreement on the default LS parity fixture (`sdofs_inf ≈ 3.6e-5`) is **not an algorithmic bug**. It is the predictable consequence of running an oversampled-DOF problem (37 DOFs vs only 25 quadpoints × 3 = 75 residual rows of limited informational rank) through a penalty LS with `constraint_weight=1.0`. The oversampled fixture (`ncoils=4, nphi=16, ntheta=8`) reaches **machine-precision CPU↔JAX parity** on the same git tree with no code change. The right response is to (a) tighten the production-scale gate to assert state parity at conservative cross-machine thresholds, (b) keep the minimal fixture as a physics-output smoke, (c) execute a series of decision-free plumbing/docs/robustness fixes that surfaced during the deepdive.

A previously proposed γ_y(φ=0, θ=0) gauge pin was **empirically refuted** under stellsym=True (the value and all 37 derivatives are structurally zero). It is withdrawn from this plan. See § Withdrawn Recommendations.

---

## Context

This document is the consolidation of a multi-round deepdive starting from the question *"how reliably does the JAX-ported BoozerSurface LS match the C++/Python reference?"* The deepdive included:

1. An 8-agent parity audit of the full JAX port (Biot-Savart, Surface Fourier, Boozer residual, integral_BdotN, SquaredFlux, BoozerSurface, M5 IFT, parity-ladder determinism). Verdict: machine-precision parity on direct kernels; tolerance parity on derivative-heavy and end-to-end LS paths.
2. A targeted 6-agent LS deepdive that mapped CPU and JAX LS solvers, verified PLU factor-sharing math, and ran an empirical CPU/JAX comparison.
3. A `newton_stab` sweep that revealed a CPU/JAX port asymmetry (CPU `run_code` does not thread the option, JAX does).
4. A gauge-pin probe that empirically refuted the proposed γ_y(0,0) fix under stellsym.
5. A null-direction probe that identified the actual cause (near-axisymmetric numerical degeneracy + weak label penalty + under-resolved quadrature).
6. Two independent review passes that corrected overstatements and validated the corrected understanding.

All empirical artifacts live under `.artifacts/_ls_deepdive_scratch/`. Hard numbers in the Appendix.

---

## Empirical Evidence (cross-machine)

### Live measurement summary, same code (cab64a15), two machines

| Quantity | Default fixture (under-sampled) | Oversampled fixture (ncoils=4, nphi=16, ntheta=8) |
|---|---|---|
| sdofs_inf | 3.6e-5 | **range across machines/trees: 1.9e-14 ↔ 3.6e-12** |
| gamma_inf | 5.2e-5 | 2.2e-14 ↔ 5.3e-12 |
| G_diff | 4.5e-6 | 0.0 (consistent across machines) |
| iota_diff | 7.5e-13 | 1.5e-17 ↔ 7.2e-16 |
| cond(H_cpu) | 5.4e+14 | 5.3e+04 (consistent) |
| cond(H_jax) | 2.2e+16 | 5.3e+04 (consistent) |
| H_inf_diff | n/a | 1.4e-12 ↔ 9.6e-10 |

**Reading:** condition number is hardware-invariant (5.3e4); state parity varies by ~2 orders of magnitude across (machine, JAX cache state, tree). The 1.9e-14 was observed on the initial probe at `cab64a15` on one machine, but the third reviewer's rerun on the current HEAD (`7822c5e0`) measured `sdofs_inf = 3.57e-12` on a different machine. Treat the 1e-14 number as a best-case observation, not a portable invariant; **gate thresholds must accommodate the worst-case measured tail (~1e-11), not the best**.

### Gauge-pin probe (stellsym=True default fixture)

```
γ(φ=0, θ=0)                   = [1.175e+00, 0.000e+00, 0.000e+00]
||∂γ_x(0,0)/∂dof||_∞          = 0.0  (across all 37 free DOFs)
||∂γ_y(0,0)/∂dof||_∞          = 0.0
||∂γ_z(0,0)/∂dof||_∞          = 0.0
Hessian smallest eigenvalues  = 1.6e-13, 3.7e-13  (rank-2 near-degeneracy)
cos(pin_grad, null_direction) = 0  for all three pin candidates
```

Proves the proposed γ_y(0,0) pin and the existing γ_z(0,0) pin are both **no-ops under stellsym**. Confirms SIMSOPT docs claim that "the z(0,0) constraint is automatically satisfied for stellarator-symmetric surfaces."

### `newton_stab` sweep (JAX-side only, CPU run_code does not thread it)

| stab | cond(H_jax) | Δsdofs (CPU↔JAX) | label_err |
|---|---|---|---|
| 0 | 2.2e+16 | 3.6e-5 | 1.1e-16 |
| 1e-12 | 9.0e+13 | 3.6e-5 | 1.1e-16 |
| 1e-8 | 8.8e+09 | 3.6e-5 | 1.1e-16 |
| 1e-4 | 8.8e+05 | 3.6e-5 | 1.1e-16 |

Confirms `cond(H+λI) ≈ σ_max/λ`. JAX cond drops linearly; CPU cond unchanged. Δsdofs static at 3.6e-5 because **CPU `run_code` does not honor `newton_stab`** (B1). After CPU pass-through, Δsdofs should track κ.

---

## Issue Inventory

| # | Severity | Issue | Status |
|---|---|---|---|
| B1 | HIGH | CPU `run_code` doesn't thread `newton_stab` into Newton polish | implemented, Wave 1 |
| C1 | HIGH | `test_ls_solve_parity` (& production-scale) gate asserts only `iota`/`label`, not `sdofs`/`G` | implemented, Wave 2 |
| D1 | LOW | CLAUDE.md references removed `hybrid` backend | implemented, Wave 1 |
| D2 | LOW | CLAUDE.md cites stale line `boozersurface_jax.py:3097` (actual: 3185-3186) | implemented, Wave 1 |
| D3 | MEDIUM | `candidate-fixed` env at `/Users/suhjungdae/code/hbt-compare/envs/...` is not runnable | implemented, Wave 1 |
| E1 | MEDIUM | `SquaredFluxJAX` does not detect surface-DOF mutation post-construction | implemented, Wave 1 |
| E3 | LOW | `group_coil_data` iteration order is Python-dict-insertion-order dependent | implemented, Wave 1 |
| C2 | MEDIUM | Coil-VJP test uses JAX-vs-JAX scalarization, not CPU oracle | implemented, Wave 3 |
| C3 | MEDIUM | Ill-conditioned exact-adjoint lane lacks action-level parity check | implemented, Wave 3 |
| B2 | INFO | scipy uses `dense-plu` host path, ondevice uses `dense-plu-shared` device path | document only |
| B3 | INFO | `optimizer_backend="ondevice"` + `least_squares_algorithm="lm"` is a different LM family than MINPACK | implemented, Wave 4 (docs) |
| E2 | LOW | `_safe_radius_squared` clamp at 1e-60 vs C++ NaN/Inf on degenerate inputs | implemented, Wave 4 (docs) |
| E4 | LOW | Traceable-bundle cache keys on `id(...)` (object-local, low practical risk) | implemented, Wave 4 |
| ~~A1~~ | ~~CRITICAL~~ | ~~γ_y(0,0) gauge pin to remove θ-rotation null direction~~ | **WITHDRAWN** (no-op under stellsym; oversampled fixture obviates) |

---

## Wave 1 — Decision-free plumbing / docs / robustness

Estimated effort: **1–2 PRs, ~150 lines total including tests.** No design debate, no measurement dependencies.

### W1.1 · B1 — Thread `newton_stab` through CPU `run_code`

**Context.** CPU `BoozerSurface.run_code` (`boozersurface.py:497-505`) calls `minimize_boozer_penalty_constraints_newton` *without* passing `stab=`. The Newton call signature has `stab=0.0` as default (`boozersurface.py:1072`). JAX `BoozerSurfaceJAX.run_code` (`boozersurface_jax.py:4713`) threads `stab=self.options["newton_stab"]` correctly. Identical option name produces different behavior across backends.

**Rationale.** Port symmetry. The option exists in `_DEFAULT_OPTIONS_LS` (`boozersurface_jax.py:3018`) and is the canonical regularization knob; the CPU side must honor the same contract or the parity testing surface is asymmetric.

**Acceptance.** Setting `options={"newton_stab": 1e-8}` must produce the same cond(H_cpu) reduction we already observe on the JAX side (κ drops linearly with stab).

**Todos:**
- [ ] Modify `boozersurface.py:497-505` to add `stab=self.options.get("newton_stab", 0.0)` to the `minimize_boozer_penalty_constraints_newton(...)` call
- [ ] Add `newton_stab` to the CPU `BoozerSurface` options accepted by `__init__` if not already (verify default propagation)
- [ ] Add CPU twin of the JAX test `test_run_code_passes_newton_stab` (locate JAX original, mirror for CPU; assert `cond(H) ∝ 1/stab` over a sweep matching `.artifacts/_ls_deepdive_scratch/ls_stab_sweep.py`)
- [ ] Re-run `.artifacts/_ls_deepdive_scratch/ls_stab_sweep.py` after the fix and confirm `cond_cpu` drops linearly with stab (matching JAX)
- [ ] Verify no existing CPU tests break (full `tests/geo/test_boozersurface.py` should pass with `stab=0` default)

### W1.2 · D1/D2/D3 — Repo-wide stale-reference cleanup

**Context.** CLAUDE.md was recently renamed env `jax-0.9.2` → `jax`, but multiple stale references remain. The third reviewer pass surfaced that the stale `hybrid` references appear in **three** lines of CLAUDE.md (lines 28, 198, 221) and the stale `boozersurface_jax.py:3097` reference appears in **three** places across the repo (`CLAUDE.md:188`, `benchmarks/_cpp_compatible_probe.py:29`, `benchmarks/_cpp_compatible_probe.py:238`) — not just in CLAUDE.md.

**Rationale.** Trust in repo documentation as a navigation aid. Every stale reference forces a verification round-trip for anyone using it as a guide. Partial cleanup is worse than no cleanup because it suggests the doc is current when only the touched lines actually are.

**Acceptance.** All cited line refs across CLAUDE.md, `benchmarks/`, and `docs/` resolve to the asserted location on current HEAD. No mentions of removed `hybrid` code path anywhere in repo.

**Todos:**
- [ ] Remove `"hybrid"` from **all** CLAUDE.md occurrences. Confirmed on current HEAD `7822c5e0`:
  - Line 28: "Private optimizer lane (`optimizer_backend="hybrid"` / `"ondevice"`)" → drop `"hybrid"`
  - Line 198: "BFGS device residency: ... `optimizer_backend="hybrid"` still depend on private line-search internals" → drop `"hybrid"`
  - Line 221: "SciPy host loop in optimizer: ... on-device and hybrid backends are now supported and validated separately" → replace with "on-device backend is now supported and validated separately"
- [ ] Update **all** stale `boozersurface_jax.py:3097` references → `boozersurface_jax.py:3185-3186` (the actual `_normalize_solver_options` stripping site; function definition starts at `boozersurface_jax.py:3122`):
  - `CLAUDE.md:188` (in "Exact Boozer scaling-limit contract" section)
  - `benchmarks/_cpp_compatible_probe.py:29` (module docstring)
  - `benchmarks/_cpp_compatible_probe.py:238` (inline comment)
- [ ] Delete or rewrite the M2 integration-tests section that references `/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python`. Either:
  - Rebuild the env from `envs/jax.yml` and update the path, OR
  - Drop the section and consolidate on `.conda/jax/bin/python` (in-tree, currently functional)
- [ ] Add a new "Floating-point reproducibility across machines" paragraph documenting the **measured cross-machine variance** on the oversampled fixture (range `1.9e-14` ↔ `3.6e-12` on `sdofs_inf` across two hardware platforms running the same source tree) so future engineers don't expect bit-identical CPU↔CPU on hardware they didn't measure on
- [ ] Verify no other line:number references across the repo are stale: `grep -rEn 'boozersurface(_jax)?\.py:[0-9]+|surfaceobjectives(_jax)?\.py:[0-9]+|optimizer_jax\.py:[0-9]+' CLAUDE.md docs/ benchmarks/` and spot-check each match resolves to the asserted symbol on current HEAD

### W1.3 · E1 — Surface-DOF fingerprint in `SquaredFluxJAX`

**Context.** `SquaredFluxJAX` (`fluxobjective_jax.py:193-204`) captures `gamma`, `normal`, `target` arrays in JIT closures at construction. Field-point and field-DOF drift are detected and raise (`:330-348`); **surface DOF mutation is NOT detected** and silently returns stale results. Reviewer confirmed with a repro.

**Rationale.** Production trap. SIMSOPT's standard idiom is `recompute_bell()` invalidation; this is a violation. The JIT closure was a deliberate Stage 2 perf choice (CLAUDE.md explicitly states "Do not call `field.set_points()` after constructing `SquaredFluxJAX`"); however that documentation is necessary but not sufficient — a misuse should raise, not silently lie.

**Acceptance.** After construction, `surface.x = surface.x + small_perturbation` followed by `flux.J()` raises `RuntimeError` (or similarly named) with a clear message pointing to the contract. A test demonstrates this.

**Todos:**
- [ ] At `SquaredFluxJAX.__init__` (`fluxobjective_jax.py:193-204`): compute `surface_dofs_fingerprint = hashlib.blake2b(surface.x.tobytes()).hexdigest()` (or `xxhash` if already in deps); store on `self._surface_dofs_fingerprint`
- [ ] Add `_raise_if_surface_dof_drifted()` helper alongside the existing `_raise_if_field_points_drifted` / `_raise_if_field_dof_layout_drifted` (`:330-348`); recompute fingerprint and compare; raise `RuntimeError("Surface DOFs mutated after SquaredFluxJAX construction; reconstruct the objective.")`
- [ ] Call the new helper at the top of `J()` (`fluxobjective_jax.py:354`) and `dJ()` (find the dJ entry point and wire it)
- [ ] Add a regression test in `tests/objectives/test_fluxobjective_jax.py` (or wherever stage-2-jax tests live): construct objective, mutate surface DOFs, assert `J()` raises with the expected message
- [ ] Verify no existing tests break (some tests may rely on the silent-staleness behavior; if so, they're incorrect and should be fixed)
- [ ] Document the new fingerprint behavior alongside the existing "Do not call `field.set_points()` after constructing" caveat

### W1.4 · E3 — Deterministic `group_coil_data` ordering

**Context.** `group_coil_data` in `jax_core/biotsavart.py:651-687` builds `by_nquad = {}` keyed on `gamma.shape[0]`, then must emit groups in the order each quadrature family first appears in the coil list. Python 3.7+ dictionaries preserve insertion order, but the parity contract should state the input-loop ordering directly rather than relying on dictionary iteration as the ordering mechanism. C++ sums coil contributions in a fixed input loop order.

**Rationale.** Future refactors could subtly change group emission order, breaking byte-parity in the mixed-quadrature lane without an obvious cause. An explicit first-input ordering eliminates the implicit dependency at near-zero cost while preserving the CPU loop's coarse summation order.

**Acceptance.** Group iteration order is independent of `by_nquad` dict insertion ordering. Existing parity tests pass.

**Todos:**
- [ ] After building `by_nquad`, replace `for indices in by_nquad.values():` with `for nquad in sorted(by_nquad.keys()): indices = by_nquad[nquad]` (or equivalent sorted iteration)
- [ ] Within each group, `indices` is already in coil-list order (insertion preserves it under append); leave the per-group order as-is
- [ ] Add a unit test in `tests/test_jax_core_biotsavart.py` (or new file) that asserts **sorted iteration order** for a single fixed mixed-nquad input. Concretely: build a coil set with quad counts `[128, 15, 15, 128]` (mixed); call `group_coil_data`; assert the returned groups are in `(qpoint_count_ascending, first_coil_index_ascending)` order; assert the per-group `indices` are in original-list ascending order. **Do not assert permutation invariance of the output:** permuting input coils legitimately changes physical coil identity and float summation order, which is not what this fix is about
- [ ] Numerical regression coverage: rely on existing field parity tests (mixed-quadrature lane already exercised by `tests/field/test_biotsavart_jax.py`) — the sort fix is structural, not numerical, so no new field-value asserts are needed
- [ ] Run the full mixed-quadrature parity suite to confirm no regressions (TF + banana coils coexist with 15-pt + 128-pt quadratures)

---

## Wave 2 — State-parity gate on oversampled fixture

Estimated effort: **1 PR, ~80 lines including tests.** Depends on Wave 1 only in spirit (Wave 1 doesn't block this).

### W2.1 · C1 — Upgrade `test_ls_solve_parity_production_scale` with state assertions

**Context.** `test_ls_solve_parity` (`test_single_stage_jax_cpu_reference.py:4718-4720`) uses the default fixture (under-sampled, κ≈1e16) and asserts only `iota_diff < 1e-6` and `label_err < 1e-3`. `test_ls_solve_parity_production_scale` (`:4722-4728`) uses the oversampled fixture (κ≈5e4) but routes through the same loose harness. The oversampled fixture is **already producing 1e-14 ↔ 1e-12 state parity** across machines, but no test is asserting on it.

**Rationale.** The existing fixture machinery already reaches near-machine precision. The only gap is the test gate. Tightening it converts a silent passage of any state drift up to 1e-3 (current) into a hard regression detector at the actual achievable cross-machine tolerance.

**Acceptance.** A new test (or expansion of the existing one) asserts the bounds below on the oversampled fixture, gated by **both** CPU and JAX Hessian condition numbers below `1e8` so it doesn't fire on degenerate sub-fixtures. Test passes on at least two machines.

**Threshold rationale.** Use **absolute** inf-norm gates (atol), not just `rtol`. `sdofs`, `G`, `iota`, and surface points all include near-zero or unit-order values where `rtol`-only would either over- or under-constrain. Measured cross-machine variance on current HEAD: `sdofs_inf ∈ [1.9e-14, 3.6e-12]` across two machines. `1e-11` provides ~2.7× headroom over the worst measured value. Tighter thresholds (`1e-12`) would be flaky across hardware; looser thresholds (`1e-10`) leave room for real regressions to slip through. `1e-11` is the cross-machine balance.

**Todos:**
- [ ] In `test_single_stage_jax_cpu_reference.py`, add a new helper `_assert_run_code_ls_state_parity(problem, ..., *, require_well_conditioned: bool)` next to `_assert_run_code_ls_parity` (`:4647`). New helper additionally asserts (all **absolute** inf-norm gates):
  - `np.max(np.abs(surf_cpu.x - surf_jax.x)) <= 1e-11`        (sdofs absolute)
  - `np.max(np.abs(surf_cpu.gamma() - surf_jax.gamma())) <= 1e-11`  (gamma points absolute)
  - `abs(G_cpu - G_jax) <= 1e-12`                              (G absolute)
  - `abs(iota_cpu - iota_jax) <= 1e-14`                        (iota absolute)
  - **Conditioning policy** (two modes via `require_well_conditioned`):
    - `require_well_conditioned=True` (production-scale tests): **hard-assert** `np.linalg.cond(res_cpu["hessian"]) < 1e8` AND `np.linalg.cond(res_jax["hessian"]) < 1e8` *as test conditions, not gates*. Rationale: the oversampled fixture is *supposed to be* well-conditioned; a regression that degrades κ above 1e8 is itself a real signal and the test should fail loudly, not silently skip.
    - `require_well_conditioned=False` (defensive helper-reuse on other fixtures): skip the state assertions with a `pytest.skip("cond(H) too high; state parity not meaningful")` message if either cond ≥ 1e8.
  - **Both-sided cond check** (independent of mode): one-sided guard would let a regression on one side degrade κ while the other stays low, hiding the disagreement. Always check both.
- [ ] Add a new test `test_ls_solve_state_parity_production_scale` that calls the helper on `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` with production options and **`require_well_conditioned=True`**. The oversampled fixture must stay well-conditioned; if it ever doesn't, that's the regression we want to surface.
- [ ] Keep the existing `test_ls_solve_parity` (default fixture, loose) as a physics smoke. Document its scope (iota+label only) with an inline comment so the next reader understands why the loose gate is intentional
- [ ] Add a docstring to `_assert_run_code_ls_state_parity` explaining the `require_well_conditioned` mode toggle, the absolute-threshold choice, and the cross-machine variance finding (`1.9e-14 ↔ 3.6e-12`). Be explicit that production-scale tests use hard-assert mode because the fixture is *supposed* to be well-conditioned, while defensive-helper mode is for unknown-κ contexts.
- [ ] Run the new test on the user's GPU (when available) under `SIMSOPT_JAX_PLATFORM=cuda` to confirm thresholds hold on hardware-2 as well; if GPU breaks 1e-11, loosen the gate to `gpu-runtime` lane's `1e-6` per `validation_ladder_contract.py`
- [ ] If practical, run on a third machine (e.g., x86 Linux) to validate the 1e-11 floor with one more data point. Update threshold to worst measured × 3 if a third machine pushes it out further.

### W2.2 · Optional: parity-ladder lane registration

**Context.** The new state-parity check could be registered as a named lane in `validation_ladder_contract.py` for discoverability.

**Todos:**
- [ ] Decide if this warrants a new ladder lane (e.g., `ls_state_parity` with rtol=1e-11). If yes, add it next to `branch-stable-resolve` (`validation_ladder_contract.py:115-133`)
- [ ] If not, just document the threshold inline in the test file

---

## Wave 3 — Oracle test coverage

Estimated effort: **2 PRs, ~150 lines.** Closes test-coverage gaps that are not currently silent-bug risks but reduce the safety net.

### W3.1 · C2 — CPU-oracle coil-VJP test (**revised after third reviewer pass**)

**Critical correction from third review.** A prior revision of this section conflated two different derivative APIs. There are **two** JAX coil-VJP entry points and two corresponding CPU oracles. The plan must pair them correctly:

| JAX function | What it differentiates | CPU oracle |
|---|---|---|
| `boozer_residual_coil_vjp` (`boozer_residual_jax.py:743`) | Boozer **residual** w.r.t. coils at **fixed surface** | `boozer_surface_residual_dB` + `B_vjp` chain (CPU equivalent of "differentiate residual through Biot-Savart") |
| `_boozer_ls_coil_vjp` (`boozersurface_jax.py:2312`) | LS penalty **gradient** w.r.t. coils (includes residual + label + z terms) | `boozer_surface_dlsqgrad_dcoils_vjp` (`surfaceobjectives.py:1486`) — docstring confirms |

The docstrings make the pairing explicit: `_boozer_ls_coil_vjp` says "Replaces CPU `boozer_surface_dlsqgrad_dcoils_vjp`"; `boozer_residual_coil_vjp` says "This replaces the CPU chain: `boozer_surface_residual_dB()` → `B_vjp()` → `sopp.biot_savart_vjp_graph()`."

**Rationale.** `test_boozer_derivatives_jax.py:822-839` validates `boozer_residual_coil_vjp` via JAX-vs-JAX scalarization. The FD oracle inside `_assert_scalar_grad_contract` (`:252-261`) IS an independent oracle, so this is not a silent-bug risk in practice. The LS-gradient VJP (`_boozer_ls_coil_vjp`) is exercised end-to-end by the M5 wrappers but does not appear to have a direct CPU-side same-state parity test today.

**Acceptance.** Two new tests, paired to the correct oracles:
- `test_residual_coil_vjp_matches_cpu_oracle`: JAX `boozer_residual_coil_vjp` vs the CPU `boozer_surface_residual_dB`-based chain at `derivative-heavy` lane tolerance (`rtol=1e-8, atol=1e-10`).
- `test_ls_coil_vjp_matches_cpu_oracle`: JAX `_boozer_ls_coil_vjp` vs CPU `boozer_surface_dlsqgrad_dcoils_vjp` at the same lane.

**Todos:**
- [ ] **Test A — residual VJP**: in `tests/geo/test_boozer_derivatives_jax.py`, add `test_residual_coil_vjp_matches_cpu_oracle` calling `boozer_residual_coil_vjp` (JAX) and the CPU chain (residual ∂B + Biot-Savart VJP). Compare per-coil cotangents at `rtol=1e-8, atol=1e-10`. Both `weight_inv_modB=True` and `False`. Use the existing same-state fixture (do NOT re-solve; both should evaluate at identical input state).
- [ ] **Test B — LS VJP**: in `tests/geo/test_boozersurface_jax.py` (or a new derivative test file under `tests/geo/`), add `test_ls_coil_vjp_matches_cpu_oracle` calling `_boozer_ls_coil_vjp(lm, booz_surf_jax, iota, G, weight_inv_modB)` and CPU `boozer_surface_dlsqgrad_dcoils_vjp(lm, booz_surf_cpu, iota, G, weight_inv_modB)` at a **converged** LS state (because LS gradient is meaningful at the converged point). Compare at `rtol=1e-8, atol=1e-10`. Both `weight_inv_modB` settings.
- [ ] Verify both CPU oracles are importable in the public env (`.conda/jax/`): `boozer_surface_dlsqgrad_dcoils_vjp` is in `simsopt.geo.surfaceobjectives` (pure Python), `boozer_surface_residual_dB` is also pure Python. Should both be available.
- [ ] If either test fails at `1e-8`, investigate: a real bug or a true machine-precision-bound case for that derivative path. Report and adjust threshold per the parity ladder, do not loosen silently.
- [ ] Cross-reference: confirm `_boozer_ls_coil_vjp` is what the M5 IFT adjoint path actually invokes (the production gradient consumer), so the parity test bites where it matters most.

### W3.2 · C3 — Action-level adjoint parity for ill-conditioned exact lane

**Context.** `validation_ladder_contract.py:103-113` (lane `exact-ill-conditioned-adjoint`) sets `vector_parity_required=False`, asserting only `residual_rel_tol`. This is correct (ill-conditioned vectors are not unique). But it leaves a coverage gap: if the adjoint solver returns total garbage, only catastrophically wrong residuals would catch it.

**Rationale.** Range-space / action-level parity is well-defined even when raw vectors aren't: `‖A·λ − b‖` (which the residual already covers) plus `‖P·λ − P·λ_ref‖` for a projector P onto a well-conditioned subspace. This catches regressions in the well-conditioned components without asserting on the genuinely-ambiguous ones.

**Acceptance.** New assertion in the `exact-ill-conditioned-adjoint` lane that verifies projected adjoint matches a CPU reference at well-conditioned-subspace tolerance.

**Todos:**
- [ ] Identify a fixture that lands in the `exact-ill-conditioned-adjoint` lane (likely an existing test marked with the lane name)
- [ ] At the converged state, compute SVD of `H = A`. Identify the well-conditioned subspace as `U_well = U[:, σ > σ_max * 1e-8]`
- [ ] Compute `λ_proj_cpu = U_well @ U_well.T @ λ_cpu` and `λ_proj_jax = U_well @ U_well.T @ λ_jax`
- [ ] Assert `‖λ_proj_cpu − λ_proj_jax‖ <= adjoint_rtol * ‖λ_proj_cpu‖` with `adjoint_rtol = 1e-6` (one order looser than `exact-well-conditioned-adjoint`)
- [ ] Add directional check: for a fixed set of deterministic test directions `v_k`, assert `|⟨v_k, λ_cpu⟩ − ⟨v_k, λ_jax⟩| <= rtol * |⟨v_k, λ_cpu⟩|`
- [ ] Document the projection definition in the lane comment

---

## Wave 4 — Optional / conditional

These items are real but lower-priority. Execute only if specific signals justify them.

### W4.1 · E2 — `_safe_radius_squared` clamp behavior

**Context.** `biotsavart.py:111` clamps `r²` at floor `1e-60` to prevent div-by-zero. C++ would return NaN/Inf on the same degenerate input. Divergent behavior on edge cases.

**Trigger to execute.** If invalid-domain testing becomes a parity concern, or if any production workflow lands on a point-on-coil configuration. Currently neither.

**Todos (conditional):**
- [ ] Decide policy: match C++ (remove clamp, accept NaN propagation) OR document divergence (preferred). Removing the clamp matches reviewer's recommendation.
- [ ] If removing: ensure no internal use of `_safe_radius_squared` relies on the floor; remove the clamp; update tests
- [ ] If documenting: add a paragraph to `docs/source/jax_acceptance.rst` noting JAX silently clamps point-on-coil while C++ NaN/Infs

### W4.2 · E4 — state-keyed traceable bundle cache key

**Original context.** `_traceable_runtime_cache_key` formerly keyed on `id(booz_jax)` and `id(bs_jax)`. The cache is stored on `booz_jax._traceable_runtime_entry_cache`, so the aliasing risk was object-local and largely theoretical, but object identity still conflated adapter lifetime with the solved/coil state that the compiled bundle actually captures.

**Trigger to execute.** If any user reports stale cache behavior tied to object lifetime patterns. Currently no signal.

**Todos (executed as root fix):**
- [x] Add explicit solve-state and coil-DOF-state tokens to `BoozerSurfaceJAX`, `BiotSavartJAX`, and `SpecBackedBiotSavartJAX`
- [x] Replace object identity in `_traceable_runtime_cache_key` with `solve_state_token`, `coil_dof_state_token`, a structural `coil_dof_extraction_spec` layout signature, and non-`id()` success-filter signatures
- [x] Advance `BiotSavartJAX` coil-state tokens from SIMSOPT ancestor DOF invalidation, not only direct adapter `x` / `full_x` writes
- [x] Verify cache hit/miss behavior is driven by runtime state, not adapter identity

### W4.3 · B3 — MINPACK-equivalent on-device LM

**Context.** `optimizer_backend="ondevice"` + `least_squares_algorithm="lm"` invokes a custom matrix-free LM with GMRES inner solve and `‖∇‖_∞`-only termination. It is NOT MINPACK `lmder`. Currently doubly opt-in.

**Trigger to execute.** If on-device LM becomes a production default, AND byte-equality (not tolerance equality) with CPU MINPACK lmder becomes a hard requirement. Currently neither.

**Todos (conditional, scoped as a research project not a patch):**
- [ ] Port MINPACK `lmder` algorithm to JAX-traceable code: pivoted QR factorization, trust-region radius management with scaling vector, three-criterion termination (`ftol`, `xtol`, `gtol`)
- [ ] Validate byte-equality on well-conditioned fixtures
- [ ] Document performance vs the existing custom-LM
- [ ] Decide deprecation policy for the existing custom-LM

---

## Withdrawn Recommendations

### A1: Add γ_y(φ=0, θ=0) gauge pin

**What was proposed.** Add a one-line penalty `0.5 * w_c * γ_y(0,0)²` symmetrically to CPU `boozersurface.py:588, 805` and JAX `boozersurface_jax.py:1881-1896`, paralleling the existing γ_z(0,0) pin, to break the alleged θ-rotation continuous symmetry and reduce κ(H) from 1e16 to 1e6-1e8.

**Why it was wrong.**

1. **Empirically inert under stellsym.** Probe (`gauge_pin_validate.py`) showed γ_y(0,0) = 0.0 exactly and `||∂γ_y(0,0)/∂dof||_∞ = 0.0` across all 37 free DOFs. The penalty contributes identically zero to objective and gradient. Same for the existing γ_z(0,0) pin; both are no-ops for stellsym=True.

2. **The diagnosis was also wrong.** I attributed the ill-conditioning to "continuous Boozer θ-rotation symmetry." Stellsym breaks the continuous θ-rotation (only θ → −θ survives, discrete). The actual cause is near-axisymmetric numerical degeneracy plus weak label penalty.

3. **The fix is unnecessary.** The oversampled fixture (`ncoils=4, nphi=16, ntheta=8`) reaches machine-precision parity (1e-14 to 1e-12 across machines) with NO code change. The ill-conditioning is a property of the under-sampled fixture, not the algorithm.

**SIMSOPT docs corroboration.** The existing z(0,0) pin is documented as "automatically satisfied for stellarator-symmetric surfaces" — i.e., a no-op for stellsym, only meaningful for non-stellsym fixtures. The empirical probe confirmed this is exact, not approximate.

**Disposition.** Recommendation fully withdrawn. No code changes proposed. Lesson preserved: distinguish "continuous symmetry in the math" (gauge-fix works) from "near-degeneracy at a discrete-symmetry limit" (requires different fixture or regularization).

### Threshold `rtol=1e-12` for state parity

**What was proposed.** Assert `np.allclose(sdofs_cpu, sdofs_jax, rtol=1e-12)` on the oversampled fixture.

**Why it was wrong.** Measured on hardware-1: `sdofs_inf = 1.88e-14`. Measured on hardware-2 (reviewer's): `sdofs_inf = 3.6e-12`. The cross-hardware floor is ~3.6e-12 not ~1e-14. `rtol=1e-12` would fail on hardware-2 even when the code is correct.

**Disposition.** Replaced with `sdofs_inf <= 1e-11` (conservative, ~2× headroom over the worst measured value). See W2.1 threshold rationale.

---

## Acceptance Criteria for the Plan as a Whole

After Wave 1 + Wave 2 land:

- [ ] `cond_cpu` responds linearly to `newton_stab` (B1 fixed)
- [ ] CLAUDE.md grep for `hybrid` returns no matches (D1)
- [ ] CLAUDE.md cited line refs all resolve correctly on current tree (D2)
- [ ] CLAUDE.md M2 integration section either points to a working env or is removed (D3)
- [ ] `SquaredFluxJAX.J()` after surface mutation raises a clear error (E1)
- [ ] `group_coil_data` iteration order is independent of dict insertion order (E3)
- [ ] `test_ls_solve_state_parity_production_scale` passes with `sdofs_inf <= 1e-11` on at least two machines (C1)
- [ ] Existing test suite passes with no new failures

After Wave 3 lands:

- [ ] `test_coil_vjp_matches_cpu_oracle` passes at `rtol=1e-8, atol=1e-10` (C2)
- [ ] Ill-conditioned exact-adjoint lane has action-level parity assertion (C3)

Wave 4 items are not part of the core acceptance. Informational completion record:

- [x] `_safe_radius_squared` clamp divergence documented in `docs/source/jax_acceptance.rst` "Domain-edge behavior" + source comment (E2)
- [x] Traceable runtime cache keys now use solved-state tokens, coil-DOF-state tokens, structural coil-layout signatures, and non-`id()` success-filter signatures instead of `id(...)` or adapter identity tokens (E4 root fix)
- [x] On-device LM vs MINPACK `lmder` algorithmic divergence documented in `src/simsopt/geo/optimizer_jax.py` module docstring + `docs/source/jax_acceptance.rst` "Optimizer family equivalence" (B3)

---

## Glossary

| Term | Definition |
|---|---|
| **LS path** | Least-squares solver path for Boozer surface: minimize `½‖r‖² + ½ w_c (label − target)²` over `[sdofs, iota, G]`. Two stages: scipy L-BFGS-B (or LM if opt-in) followed by hand-rolled Newton polish. |
| **Exact path** | Newton on residual=0 with Lagrange multipliers for label constraints (volume / area / toroidal flux). Different solver entirely (`solve_residual_equation_exactly_newton`). |
| **Decision vector** | `[surface_dofs, iota, G?]`. `G` is included when `optimize_G = (G is not None)`. |
| **κ (kappa)** | Condition number `σ_max(H) / σ_min(H)`. Industry rule: `log10(κ)` digits of double-precision are lost. |
| **Null direction** | Eigenvector of `H` with near-zero eigenvalue; direction in DOF space where the objective is locally flat. |
| **Gauge ambiguity** | Continuous symmetry in the parameterization that makes the parameter representation non-unique (only the *physical* output is unique). |
| **Stellsym** | Stellarator symmetry. Discrete `θ → −θ, φ → −φ` symmetry that halves the DOF count. **Breaks** the continuous Boozer θ-rotation. |
| **`weight_inv_modB`** | Boolean flag in the residual definition. When True, `r = (1/|B|) · (G·B − |B|²·(∂γ/∂φ + iota·∂γ/∂θ))`. Default True. |
| **`newton_stab`** | Tikhonov damping for the Newton polish: `dx = (H + stab·I)^{-1} ∇f`. Default 0. Currently honored on JAX side, not CPU side (B1). |
| **`constraint_weight`** | Penalty weight `w_c` on the label-error term. Default 1.0. Larger values enforce the label more strictly at the cost of conditioning in label-orthogonal directions. |
| **Parity ladder** | SSOT in `benchmarks/validation_ladder_contract.py` defining per-lane rtol/atol contracts. Lanes: `direct-kernel`, `derivative-heavy`, `exact-well-conditioned-adjoint`, `exact-ill-conditioned-adjoint`, `branch-stable-resolve`, `fd-gradient`, `gpu-runtime`, etc. |
| **Strict gate** | `_pre_newton_census_gate_failures` in `single_stage_init_parity.py:2110-2155`. Byte-identity contract between CPU and JAX at pre-Newton snapshot. |
| **PLU factor-sharing** | LS-lane bit-equal forward/adjoint Hessian action by reusing the same `(lu, piv)` packed factors under `lax.stop_gradient`. See PLU agent finding for the 4-clause precondition. |

---

## References

### Code locations (current tree)

| Reference | Location |
|---|---|
| CPU `run_code` LS path | `src/simsopt/geo/boozersurface.py:441-508` |
| CPU `run_code` Newton call without `stab=` (B1) | `src/simsopt/geo/boozersurface.py:497-505` |
| CPU `minimize_boozer_penalty_constraints_newton` | `src/simsopt/geo/boozersurface.py:1065-1200` |
| CPU γ_z(0,0) penalty | `src/simsopt/geo/boozersurface.py:588, 805` |
| CPU LS-gradient VJP — oracle for `_boozer_ls_coil_vjp` (W3.1 Test B) | `src/simsopt/geo/surfaceobjectives.py:1486` (`boozer_surface_dlsqgrad_dcoils_vjp`) |
| CPU residual-derivative chain — oracle for `boozer_residual_coil_vjp` (W3.1 Test A) | `src/simsopt/geo/surfaceobjectives.py:boozer_surface_residual_dB` + Biot-Savart VJP path |
| JAX fixed-surface residual-VJP | `src/simsopt/geo/boozer_residual_jax.py:743` (`boozer_residual_coil_vjp`) |
| JAX LS-gradient VJP | `src/simsopt/geo/boozersurface_jax.py:2312` (`_boozer_ls_coil_vjp`) |
| JAX `run_code` LS path | `src/simsopt/geo/boozersurface_jax.py:4985-5161` |
| JAX `newton_stab` threading | `src/simsopt/geo/boozersurface_jax.py:4713` |
| JAX `_DEFAULT_OPTIONS_LS` | `src/simsopt/geo/boozersurface_jax.py:3010-3024` |
| JAX γ_z(0,0) penalty | `src/simsopt/geo/boozersurface_jax.py:1881-1896` |
| JAX optimizer backends | `src/simsopt/geo/optimizer_jax.py:108` (`VALID_OPTIMIZER_BACKENDS = {scipy, ondevice}`) |
| JAX LS algorithms | `src/simsopt/geo/optimizer_jax.py:121` (`VALID_LEAST_SQUARES_ALGORITHMS = {quasi-newton, lm}`) |
| Default `least_squares_algorithm` | `src/simsopt/geo/optimizer_jax.py:667, 731` (`"quasi-newton"`) |
| `SquaredFluxJAX` closure capture (E1) | `src/simsopt/objectives/fluxobjective_jax.py:193-204` |
| `group_coil_data` (E3) | `src/simsopt/jax_core/biotsavart.py:651-687` |
| `_safe_radius_squared` clamp (E2) | `src/simsopt/jax_core/biotsavart.py:111` |
| Traceable runtime cache key (E4) | `src/simsopt/geo/surfaceobjectives_jax.py:4120-4144, 4168-4170` |
| Stale `3097` references (W1.2) | `CLAUDE.md:188`, `benchmarks/_cpp_compatible_probe.py:29`, `benchmarks/_cpp_compatible_probe.py:238` |
| Stale `hybrid` references (W1.2) | `CLAUDE.md:28, 198, 221` |
| Parity ladder SSOT | `benchmarks/validation_ladder_contract.py:52-210` |
| Strict gate | `benchmarks/single_stage_init_parity.py:2110-2155` |
| LS parity fixture builder | `benchmarks/benchmark_problem.py:103-170` |
| Default-fixture LS parity test | `tests/integration/test_single_stage_jax_cpu_reference.py:4710-4720` |
| Oversampled LS parity test (C1) | `tests/integration/test_single_stage_jax_cpu_reference.py:4722-4728` |
| `_assert_run_code_ls_parity` harness | `tests/integration/test_single_stage_jax_cpu_reference.py:4647-4707` |

### External documentation

| Topic | Source |
|---|---|
| BoozerSurface API + stellsym z(0,0) auto-satisfaction | <https://simsopt.readthedocs.io/v1.8.0/simsopt_user.geo.html> |
| SciPy `least_squares(method="lm")` semantics (MINPACK lmder wrapper) | <https://scipy.github.io/devdocs/reference/generated/scipy.optimize.least_squares.html> |
| JAX `lu_solve` (forward/transpose flag) | <https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.lu_solve.html> |
| Boozer-surface formulation | arXiv:2203.03753 |

### Empirical artifacts (in-tree)

| Artifact | Purpose |
|---|---|
| `.artifacts/_ls_deepdive_scratch/ls_parity_harness.py` | CPU vs JAX LS converged state on default fixture |
| `.artifacts/_ls_deepdive_scratch/ls_stab_sweep.py` | `newton_stab` sweep showing cond ∝ 1/stab on JAX side |
| `.artifacts/_ls_deepdive_scratch/gauge_pin_validate.py` | γ_y / γ_z(0,0) structural-zero probe under stellsym |
| `.artifacts/_ls_deepdive_scratch/null_direction_probe.py` | Eigendecomposition of converged Hessian; null-direction Fourier-mode analysis |
| `.artifacts/_ls_deepdive_scratch/oversampled_parity.py` | Default vs oversampled fixture state-parity comparison |
| `.artifacts/_ls_deepdive_scratch/reproduce_oversampled.py` | Triple-run reproducibility check on oversampled fixture |

---

## Appendix A — Raw harness outputs

### A.1 Default fixture (under-sampled, this machine)

```
=== default (ncoils=2, nphi=5, ntheta=5, mpol=ntor=2) ===
  surface DOFs:        37 free
  Hessian size:        (39, 39)
  sdofs_inf:           3.615065e-05
  gamma_inf (surface): 5.176195e-05
  iota_diff:           7.535812e-13
  G_diff:              4.472881e-06
  ||r_cpu||:           3.911330e-13
  ||r_jax||:           1.004133e-13
  cond(H_cpu):         5.430212e+14
  cond(H_jax):         2.238868e+16
```

### A.2 Oversampled fixture (this machine, three independent runs)

```
[run #1, #2, #3 — bit-identical]
  sdofs_inf      = 1.883129e-14
  gamma_inf      = 2.198242e-14
  iota_diff      = 1.537704e-17
  G_diff         = 0.000000e+00
  cond(H_cpu)    = 5.277221e+04
  cond(H_jax)    = 5.277221e+04
  H_inf_diff     = 1.350919e-12
  H_rel_diff     = 8.309641e-14
```

### A.3 Oversampled fixture (reviewer's machine, per their report)

```
  sdofs_inf      = 3.571e-12
  gamma_inf      = 5.339e-12
  iota_diff      = 7.159e-16
  G_diff         = 0
  cond(H_cpu)    = 5.277e+04
  cond(H_jax)    = 5.277e+04
  H_inf_diff     = 9.6e-10  (entrywise, not bit-identical)
```

### A.4 Gauge-pin probe (γ_y/γ_z(0,0) structural-zero check, stellsym=True)

```
surface stellsym = True
surface mpol = 2, ntor = 2, nfp = 2
surface ndofs (free) = 37

γ(φ=0, θ=0) = [1.17511314 0. 0.]
  γ_x(0,0)              = 1.175113e+00
  γ_y(0,0)              = 0.000000e+00
  γ_z(0,0)              = 0.000000e+00
  ||∂γ_x(0,0)/∂dof||_∞  = 0.000000e+00  (zero across all 37 DOFs)
  ||∂γ_y(0,0)/∂dof||_∞  = 0.000000e+00
  ||∂γ_z(0,0)/∂dof||_∞  = 0.000000e+00

Hessian eigenvalue spectrum (low to high):
  λ[0] = 1.561697e-13   ← near-zero (rank-deficient)
  λ[1] = 3.679294e-13   ← near-zero
  λ[2] = 2.502871e-05
  λ[3] = 1.181669e-04
  ...
  λ[38] = 8.791865e+01

cos(γ_y(0,0)_grad, null_direction) = 0.000000e+00  ← pin orthogonal to null
```

### A.5 `newton_stab` sweep (JAX side; CPU does not respond because B1)

```
stab        cond(H_cpu)    cond(H_jax)    Δsdofs    Δlabel_jax
0e+00       5.43e+14       2.24e+16       3.6e-05   1.1e-16
1e-12       5.43e+14       8.98e+13       3.6e-05   1.1e-16
1e-10       5.43e+14       8.79e+11       3.6e-05   1.1e-16
1e-08       5.43e+14       8.79e+09       3.6e-05   1.1e-16
1e-06       5.43e+14       8.79e+07       3.6e-05   1.1e-16
1e-04       5.43e+14       8.79e+05       3.6e-05   1.1e-16
```

Note: `cond(H_jax)` scales as `σ_max(H)/stab ≈ 8.8e1 / stab`. Matches Tikhonov theory exactly. `cond(H_cpu)` is unchanged — confirms CPU `run_code` is not threading `newton_stab` (B1).

---

## Provenance

**Deepdive driven by:** user prompt "deploy 4.7 opus max subagents. goal: how much are the jax ported code reliably compare to existing cpp/python code? Rely on math, physics, computation, not artifacts."

**Iteration history:**
1. 8-agent broad audit → verdict "machine-precision on direct kernels, tolerance parity on derivatives"
2. 6-agent LS deepdive → identified default fixture's 1e-5 sdofs drift, cond=1e16
3. Original gauge-pin recommendation (A1) — withdrawn after empirical probe
4. First reviewer pass — caught A1 no-op under stellsym; reframed conditioning as near-degeneracy + fixture choice
5. My validation of first reviewer (oversampled fixture gives 1e-14 on this machine)
6. Second reviewer pass — caught cross-machine variance (1e-14 ↔ 1e-12); corrected MINPACK/quasi-newton wording; corrected threshold `1e-12 → 1e-11`
7. First version of this document
8. **Third reviewer pass** (this revision): caught **W3.1 oracle mismatch** (had paired `boozer_residual_coil_vjp` with the wrong CPU oracle — corrected to two separate tests, one per derivative API); caught **stale doc refs broader than CLAUDE.md** (`benchmarks/_cpp_compatible_probe.py` also has stale `3097` refs; CLAUDE.md has `hybrid` on lines 28, 198, 221 not just 28); caught **HEAD-vs-snapshot drift** (current HEAD `7822c5e0` ≠ snapshot `cab64a15`); recommended **absolute thresholds + dual cond guard** for W2.1 instead of `rtol=1e-12`; caught **E3 test over-assertion** (permutation invariance can legitimately break — restrict to single-fixed-input sort assertion).

**Net correction (cumulative):** the original "machine precision parity" reading was correct in spirit but specific numbers are machine- and tree-dependent. Plan now uses conservative cross-machine absolute thresholds with dual cond-number guard, and correctly pairs JAX↔CPU derivative APIs for the oracle tests.
