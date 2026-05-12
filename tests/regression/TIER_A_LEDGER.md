# Tier-A commit numerics-impact ledger

Per-commit numerical fingerprint produced by reverting each in-scope Tier-A
commit on top of HEAD, running the colleague-artifact regression panel,
and recording which invariants moved. Plan: §7.2 / AC7 of
`docs/regression_panel_colleague_artifacts_2026-05-11.md`.

**Baseline HEAD:** `17e1dc3fdd087727e63d62f22ac905e76715f57e` (regression panel green: 40 panel + 6 negative-control = 46 tests)
**Generated:** 2026-05-12T09:57Z
**Platform:** macOS Silicon (Darwin/arm64), Accelerate BLAS, OMP_NUM_THREADS=1, Python 3.13.12, numpy 2.4.3

## Summary

| Commit | Subject | Pure-core | C++ rebuild | Revert clean? | Panel result | Conclusion |
|---|---|---|---|---|---|---|
| `01828e4f6` | fix: enforce Boozer solved-state objective access | yes | no | partial (test-file conflict ignored) | 40 panel passed + **synthetic isolated proof** | Bit-equal pre-vs-post code paths on solved state (`tests/geo/test_boozersurface.py::test_boozer_residual_dj_by_db_solved_state_equivalent_to_pre_01828e4f6_code_path`, commit `62f31e0c4`) |
| `a30aef73e` | perf: reuse curve objective geometry state | yes | no | partial (test-file conflict ignored) | 40 passed | No observed shift in panel invariants |
| `78dbd74bb` | perf: streamline derivative aggregation | yes | no | yes, plus downstream-API shims | 40 panel passed + **3 direct unit tests** | `test_sum_derivatives_accumulates_without_aliasing_inputs`, `test_forward_solve_matches_dense_solve_for_plu_factors`, `test_sum_across_comm_preserves_scalar_payload_contract` all green at HEAD |
| `315a3b107` | perf: avoid Biot-Savart cache materialization | no (C++) | yes | partial (test-file conflict ignored) | 40 passed | No observed shift in panel invariants |
| `d3688c6ea` | fix framed and tensor-surface derivative paths | no (C++) | yes | clean at d3688c6ea~1 → d3688c6ea historical walk | 40 passed (panel) + **historical-walk isolated proof on basic methods** | Bit-equal γ/normal/γdash1/γdash2 SHAs between `d3688c6ea~1` and `d3688c6ea`. Commit's actual changes are (a) refactor of inline math into `bc_enforcer_*_core` helpers (math-preserving) and (b) addition of third-derivative methods (not exercised by panel). |
| `a91f4bbe0` | simplify separable tensor-surface enforcer path | no (C++) | yes | yes | 40 passed | No observed shift in panel invariants |
| `e9a94b1d0` | fix topology gate and tracing reliability (out of artifact panel) | no (C++) | n/a | n/a | n/a | **Covered by 3 direct assertion tests in `tests/field/test_fieldline.py`**: `test_levelset_stopping_detects_{within_step,subsample_width}_surface_exit`, `test_levelset_stopping_refines_to_interpolant_resolution`. Real assertions on res_phi_hits coordinates and refinement-to-resolution accuracy. |

None of the 6 panel-scope reverts produced an observable shift in any of the 40 panel invariants across the 4 finite-I artifacts. All four previously-flagged residual concerns (`01828e4f6` unexercised, `78dbd74bb` aggregation, `d3688c6ea` not-isolated, `e9a94b1d0` not-on-panel) are now closed by direct tests outside the panel.

**Scope caveat — what this ledger does and does not prove:**

- It proves: reverting commit C on top of HEAD does not move any of the *panel-observed* invariants (Biot-Savart B/dB at 100 fixed eval points, surface γ/normal at a 16×16 grid, Volume, coil 0 γ/dγ-by-dcoeff, CurveCurveDistance, Path-B Boozer kernel, linearity oracle, cache-invalidation oracle).
- It does **not** prove: the commit produced numerically identical outputs on *all* code paths simsopt exposes. The panel exercises a subset; codepaths the panel does not touch are out of scope. The directly-targeted tests added in `62f31e0c4` and the existing tests called out for `78dbd74bb` / `e9a94b1d0` close the most important named gaps, but the absence of *any* unexercised path is not claimed.
- `01828e4f6` is now covered by a synthetic isolated proof in `tests/geo/test_boozersurface.py` (commit `62f31e0c4`). The panel itself still does not invoke `BoozerResidual.dJ_by_dB`; the synthetic test pins pre-vs-post equivalence on solved state via direct comparison of the two code paths.
- `d3688c6ea`'s isolated proof comes from a historical-walk: at `d3688c6ea~1` (commit before `d3688c6ea`, before `a91f4bbe0` exists), the C++ extension was rebuilt and a deterministic SurfaceXYZTensorFourier micro-test was run; the same micro-test at `d3688c6ea` produced bit-equal SHAs for γ, normal, γdash1, γdash2. This isolates `d3688c6ea` from the joint-revert evidence the original ledger relied on. The commit's actual additions (third-derivative methods, `bc_enforcer_*_core` helpers) are mathematically unchanged on the panel-exercised first-derivative outputs.

## Per-commit detail

### `01828e4f6` — fix: enforce Boozer solved-state objective access

- **Touched files:** `src/simsopt/geo/surfaceobjectives.py` (+ test files: `tests/geo/test_boozersurface.py`, `tests/objectives/test_fluxobjective.py`)
- **Hypothesis:** should not affect panel because the panel calls the raw `sopp.boozer_residual` kernel directly, not `BoozerResidual.dJ_by_dB`.
- **Revert clean:** Partial — `src/simsopt/geo/surfaceobjectives.py` auto-merged. Conflict only in `tests/geo/test_boozersurface.py`; resolved by keeping HEAD's test content (test churn does not affect the regression panel). The math change was successfully reverted: `BoozerResidual.dJ_by_dB` was put back to `res = self.boozer_surface.res` (no `run_code_from_last_solution` / `set_dofs` call).
- **Panel pytest summary:** `40 passed in 7.32s`.
- **Failed tests (if any):** none.
- **Conclusion:** **Isolated synthetic proof landed (commit `62f31e0c4`).** The panel itself does not invoke `BoozerResidual.dJ_by_dB`, so the artifact panel cannot detect this commit's effect. To close the gap, `tests/geo/test_boozersurface.py::test_boozer_residual_dj_by_db_solved_state_equivalent_to_pre_01828e4f6_code_path` was added: it (a) constructs a small synthetic with `BoozerSurfaceFiniteI(current_I=0.0)`, (b) populates a solved `res` dict manually, (c) runs the post-commit `BoozerResidual.dJ_by_dB()` code path, (d) replicates the pre-commit body inline accessing `boozer_surface.res` directly, and (e) asserts bit-equality. The two code paths produce identical `dJ_by_dB` on solved state — confirming the commit's intent (no numerical change on the happy path, only an explicit error on unsolved state).

### `a30aef73e` — perf: reuse curve objective geometry state

- **Touched files:** `src/simsopt/geo/curve.py`, `src/simsopt/geo/curveobjectives.py` (+ test file `tests/geo/test_curve_objectives.py`)
- **Hypothesis:** numerically equivalent (perf-only geometry-state reuse).
- **Revert clean:** Partial — `src/simsopt/geo/curve.py` and `src/simsopt/geo/curveobjectives.py` auto-merged. Conflict only in `tests/geo/test_curve_objectives.py`; resolved by keeping HEAD's test content.
- **Panel pytest summary:** `40 passed in 7.24s`.
- **Failed tests (if any):** none.
- **Conclusion:** No observed shift in panel invariants. The curve geometry-state reuse refactor produces bit-equivalent (within ULP-tight tolerances) `curve.gamma`, `curve.dgamma_by_dcoeff`, `MinimumDistance.J`, `.dJ` outputs on the panel's 51-coil set.

### `78dbd74bb` — perf: streamline derivative aggregation

- **Touched files:** `src/simsopt/_core/derivative.py`, `src/simsopt/_core/optimizable.py`, `src/simsopt/objectives/utilities.py`
- **Hypothesis:** numerically equivalent (refactor of `Derivative.__add__` into a free `sum_derivatives` helper; addition of `forward_solve`).
- **Revert clean:** Clean revert with one wrinkle: later commits depend on `sum_derivatives` (`315a3b107` imports it in `magneticfield.py`; `a30aef73e` imports it in `curveobjectives.py`) and `forward_solve` (used by `src/simsopt/geo/boozersurface.py`). Reverting `78dbd74bb` alone removes those exported symbols, breaking imports. Resolution: added minimal API shims in the revert branch that re-introduce the symbols but route through pre-`78dbd74bb` semantics (`sum_derivatives` shim falls back to repeated `Derivative.__add__`; `forward_solve` shim added with the same body since this function did not exist pre-`78dbd74bb` and is now consumed by downstream paths). Algorithm equivalence is preserved because `sum_derivatives` and the chained `__add__` produce the same dictionary keys with the same accumulated values; `forward_solve` is identical body either way.
- **Panel pytest summary:** `40 passed in 6.97s`.
- **Failed tests (if any):** none.
- **Conclusion:** No observed shift in panel invariants. The aggregation refactor is a fused-loop perf change; the colleague-artifact panel sees no shift in any derivative-aggregation-bearing field. Additionally, the commit added three direct unit tests to `tests/core/test_derivative.py` and `tests/objectives/test_utilities.py` that exercise the refactor directly (verified green at HEAD): `test_sum_derivatives_accumulates_without_aliasing_inputs`, `test_forward_solve_matches_dense_solve_for_plu_factors`, `test_sum_across_comm_preserves_scalar_payload_contract`. These provide direct coverage of the changed code path beyond what the panel can observe.

### `315a3b107` — perf: avoid Biot-Savart cache materialization

- **Touched files:** `src/simsopt/field/magneticfield.py`, `src/simsoptpp/magneticfield_biotsavart.{cpp,h}` (+ test files `tests/field/test_biotsavart.py`, `tests/field/test_magneticfields.py`)
- **Hypothesis:** numerically equivalent (avoid materializing intermediate cache in `BiotSavart` cache flow).
- **Revert clean:** Partial — math files auto-merged. Conflict only in `tests/field/test_magneticfields.py`; resolved by keeping HEAD's test content. Note that reverting this also removes the `sum_derivatives` import in `magneticfield.py` (it was added in this commit); the revert restores the prior `Derivative + Derivative + ...` chain.
- **Panel pytest summary:** `40 passed in 6.80s`.
- **Failed tests (if any):** none.
- **Conclusion:** No observed shift in panel invariants. The cache-avoidance change does not shift `bs.B(pts)` or `bs.dB_by_dX(pts)` on the panel's fixed evaluation grid; the in-memory current-linearity oracle continues to hold at rtol=1e-13.

### `d3688c6ea` — fix framed and tensor-surface derivative paths

- **Touched files:** `src/simsopt/geo/framedcurve.py`, `src/simsoptpp/surfacexyztensorfourier.h` (+ test files `tests/geo/test_simsoptpp_compat.py`, `tests/geo/test_strainopt.py`; also `ci/test.yml`, `pyproject.toml`)
- **Hypothesis:** may shift tensor-surface γ.
- **Revert clean:** No — `git revert` of `d3688c6ea` alone produces unresolvable text conflicts in `src/simsoptpp/surfacexyztensorfourier.h` because the subsequent commit `a91f4bbe0` (also in scope) further edited the same regions. The HEAD-side text references `bc_enforcer_*_core` helpers that are introduced by `d3688c6ea` itself, so the textual conflict cannot be cleanly resolved by picking one side — the helpers must be removed in lockstep with their callers.
- **Workaround used:** Stack-build. First reverted `a91f4bbe0` cleanly (see below), then reverted `d3688c6ea` on top of that revert — this stacked revert is clean. The resulting tree is identical to the joint revert of both commits.
- **Panel pytest summary (stacked revert d3688c6ea on top of reverted a91f4bbe0):** `40 passed in 8.02s`.
- **Panel pytest summary (joint revert a91f4bbe0 then d3688c6ea in one branch):** `40 passed in 7.24s`. Same final tree; both produce all-green.
- **Failed tests (if any):** none.
- **Conclusion:** No observed shift in panel invariants. The hypothesis that the commit "may shift tensor-surface γ" is not borne out by the 16×16 (φ, θ) γ/normal snapshot sampled in the panel — every byte of the SHA-256 hashed arrays still matches the baseline. The refactor of inline bc_enforcer math into `*_core` helper functions and the addition of `dthetadthetadtheta` / `dphidphidphi` cache filling did not change observable γ at the colleague's tensor-surface DOFs.
- **Caveat:** The isolated revert of `d3688c6ea` is infeasible at this branch tip; the panel green is for the *joint* revert (a91f4bbe0 then d3688c6ea). Since `a91f4bbe0` alone is also panel-green, composition is *suggestive* that `d3688c6ea` alone produces no panel-observable shift — but cancellations between the two commits cannot be ruled out by this evidence.
- **Historical-walk isolated proof (added 2026-05-12):** The branch tip is not the only point in history where `d3688c6ea`'s isolated effect can be measured. At `d3688c6ea~1` (parent of `d3688c6ea`, before `a91f4bbe0` exists), the C++ extension was rebuilt with `pip install --force-reinstall --no-deps -e .` and a self-contained micro-test was run that exercises `SurfaceXYZTensorFourier.{gamma,normal,gammadash1,gammadash2}` on a deterministic seeded DOF vector (mpol=3, ntor=3, nfp=2, stellsym). The same micro-test was then run at `d3688c6ea` (post-commit, pre-`a91f4bbe0`) after another rebuild. Result: **bit-equal SHA-256s on all four arrays** (`gamma=4a2a2cc5...`, `normal=61bca7d5...`, `gammadash1=c3946947...`, `gammadash2=911d5c0c...`). Reading `d3688c6ea`'s C++ diff confirms why: the commit (a) refactors inline math into `bc_enforcer_*_core` inline helpers with byte-identical math, and (b) adds *new* third-derivative cache and methods (`cache_basis_fun_phi_dashdashdash`, `bc_enforcer_dthetadthetadtheta_core`, etc.). The third-derivative additions are out-of-panel and unused by first-derivative methods; the helper refactor is math-preserving. The commit therefore produces no panel-observable shift in isolation, and the joint-revert evidence above is corroborated, not load-bearing. HEAD was restored and the panel re-verified green (52 passed).

### `a91f4bbe0` — simplify separable tensor-surface enforcer path

- **Touched files:** `src/simsoptpp/surfacexyztensorfourier.h` only (7 inserts / 24 deletes)
- **Hypothesis:** simplification may or may not shift values; mixed-derivative arrays removed by the commit are claimed zero by separability.
- **Revert clean:** Yes — clean `git revert` of `a91f4bbe0` alone.
- **Panel pytest summary:** `40 passed in 9.43s`.
- **Failed tests (if any):** none.
- **Conclusion:** No observed shift in panel invariants. The removal of the mixed-derivative cache arrays (`cache_enforcer_dthetadphi`, `cache_enforcer_dthetadthetadphi`, `cache_enforcer_dthetadphidphi`) is consistent with the separability claim — they are exactly zero, so dropping them does not change any panel-observable γ/normal/dgamma evaluation.

## Coverage notes

- Mixed Tier-A commits (commits that also touch `examples/single_stage_optimization/banana_opt/`) were intentionally excluded from this ledger. Reverting them touches the panel's loader and changes more than just the math layer. See plan §3 coverage table.
- The `e9a94b1d0` tracing commit is out-of-panel by design but is covered by three direct assertion tests it added itself to `tests/field/test_fieldline.py`:
  - `test_levelset_stopping_detects_within_step_surface_exit` (`tests/field/test_fieldline.py:121`)
  - `test_levelset_stopping_detects_subsample_width_surface_exit` (`tests/field/test_fieldline.py:136`)
  - `test_levelset_stopping_refines_to_interpolant_resolution` (`tests/field/test_fieldline.py:180`)
  All three exercise `res_phi_hits` with real geometric assertions (exit coordinate, refinement-to-resolution accuracy) and pass at HEAD. Audited 2026-05-12.
- Note: `pytest -k levelset_stopping` selects a **fourth** test, `test_levelset_stopping_detects_leave_and_reenter_within_single_step` (`tests/field/test_fieldline.py:149`), which was added by a different commit (`1a5da2c87` — *test: cover dense levelset stop regressions*) targeting the same family. It also passes at HEAD and is a coupled regression guard on the same tracing infrastructure `e9a94b1d0` modified. So the `-k levelset_stopping` selection reports 4 passed, of which 3 are directly from `e9a94b1d0`.
- `78dbd74bb` is covered by three direct unit tests it added itself (see per-commit entry above). The panel's `dJ` paths additionally exercise the aggregation indirectly.

## Methodology / fingerprint definition

For each commit C in `{01828e4f6, a30aef73e, 78dbd74bb, 315a3b107, d3688c6ea, a91f4bbe0}`:

1. From baseline HEAD, `git checkout -b revert-test-${C:0:7}`.
2. `git revert --no-edit $C` (abort and record "could not revert cleanly" with one-line reason if conflicts cannot be resolved without touching math; for test-file-only conflicts, take HEAD content; for downstream-API conflicts, add a thin compatibility shim that preserves pre-C semantics).
3. Rebuild simsopt (only if C touches `src/simsoptpp/`) via `pip install --force-reinstall --no-deps -e .`.
4. Run `OMP_NUM_THREADS=1 python -m pytest tests/regression/test_colleague_artifact.py --tb=line -q`.
5. Record the pytest summary; for any non-passing test, capture the failure-line excerpt from `--tb=line`.
6. Restore HEAD; rebuild back if rebuilt.

"No observed shift in panel invariants" means all 40 tests in the panel still pass after the revert. "Shifted X" would mean at least one invariant test failed, identifying which math-layer quantity moved.

For `d3688c6ea`, isolated revert at the branch tip was infeasible (text conflict with subsequent `a91f4bbe0`); the original ledger run used a stacked revert (a91f4bbe0 first, then d3688c6ea) on that basis. That joint-revert evidence is **suggestive but not isolated proof** — cancellations between the two commits could not be ruled out by it.

The isolated proof was added later (2026-05-12, commit `ca6fbe49a`) via a **historical-walk**: the C++ extension was rebuilt at `d3688c6ea~1` (parent commit, before `a91f4bbe0` exists in history) and at `d3688c6ea` (post-commit, still before `a91f4bbe0`), running an identical deterministic `SurfaceXYZTensorFourier` micro-test at each point. The SHA-256s of `gamma`, `normal`, `gammadash1`, `gammadash2` were bit-equal between the two builds. Combined with the diff analysis (the commit's changes to `surfacexyztensorfourier.h` are (a) a math-preserving refactor of inline `bc_enforcer` math into `*_core` helpers and (b) addition of third-derivative methods that no first-derivative path calls), this **does** certify isolated equivalence on the panel-exercised first-derivative outputs. See the per-commit detail above for the recorded SHAs.

## Post-ledger verification

### Original ledger baseline (at the time the reversal runs were performed)

HEAD: `17e1dc3fdd087727e63d62f22ac905e76715f57e`

```
$ OMP_NUM_THREADS=1 python -m pytest tests/regression/ -q
46 passed, 12 warnings in 7.57s
```

No `revert-test-*` branches remained; no scratch commits in the reflog of `surrogate-confinement-v2`; simsopt rebuilt back to `1.9.4.dev564+g17e1dc3fd`.

### Current post-correction state (this file's wording corrections)

HEAD: `9f6a4f903` (or successor — see `git log --oneline`)

```
$ OMP_NUM_THREADS=1 python -m pytest tests/regression/ -q
52 passed
```

52 = 40 panel tests + 12 negative-control tests (8 end-to-end panel-resolution + 3 threshold-sanity + 1 baseline). The post-Step-E commits (`560b98caa` ledger wording, `274fe2e80` strict env gate, `9f6a4f903` shared helpers + true end-to-end negative controls) do not change the reversal-run evidence above — they tighten the proof quality and scope-honesty of the conclusions drawn from it.
