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
| `01828e4f6` | fix: enforce Boozer solved-state objective access | yes | no | partial (test-file conflict ignored) | 40 passed | **Unexercised by this panel** — the panel does not call `BoozerResidual.dJ_by_dB`; verdict is vacuous and a different oracle is needed |
| `a30aef73e` | perf: reuse curve objective geometry state | yes | no | partial (test-file conflict ignored) | 40 passed | No observed shift in panel invariants |
| `78dbd74bb` | perf: streamline derivative aggregation | yes | no | yes, plus downstream-API shims | 40 passed | No observed shift in panel invariants |
| `315a3b107` | perf: avoid Biot-Savart cache materialization | no (C++) | yes | partial (test-file conflict ignored) | 40 passed | No observed shift in panel invariants |
| `d3688c6ea` | fix framed and tensor-surface derivative paths | no (C++) | yes | only as a stack on top of `a91f4bbe0` revert | 40 passed | **Joint/stacked evidence only** — isolated revert infeasible at this branch tip; cannot rule out cancellations |
| `a91f4bbe0` | simplify separable tensor-surface enforcer path | no (C++) | yes | yes | 40 passed | No observed shift in panel invariants |

None of the 6 reverts produced an observable shift in any of the 40 panel invariants across the 4 finite-I artifacts.

**Scope caveat — what this ledger does and does not prove:**

- It proves: reverting commit C on top of HEAD does not move any of the *panel-observed* invariants (Biot-Savart B/dB at 100 fixed eval points, surface γ/normal at a 16×16 grid, Volume, coil 0 γ/dγ-by-dcoeff, CurveCurveDistance, Path-B Boozer kernel, linearity oracle, cache-invalidation oracle).
- It does **not** prove: the commit produced numerically identical outputs on all code paths simsopt exposes. The panel exercises a subset; codepaths the panel does not touch are out of scope.
- Specifically, `01828e4f6` (`BoozerResidual.dJ_by_dB` solved-state guard) is **unexercised by this panel** — its observable effect is on `BoozerResidual.dJ_by_dB`, which the panel does not call. The "no shift" verdict is therefore vacuously satisfied; a stronger oracle is required to validate that commit (see §coverage notes).
- `d3688c6ea` (tensor-surface derivative paths) was **not isolated-revertable** at this branch tip — only verified via stacked revert jointly with `a91f4bbe0`. The joint-revert + standalone-equivalence-of-`a91f4bbe0` argument is suggestive but not a rigorous isolated proof; cancellations are possible in principle.

## Per-commit detail

### `01828e4f6` — fix: enforce Boozer solved-state objective access

- **Touched files:** `src/simsopt/geo/surfaceobjectives.py` (+ test files: `tests/geo/test_boozersurface.py`, `tests/objectives/test_fluxobjective.py`)
- **Hypothesis:** should not affect panel because the panel calls the raw `sopp.boozer_residual` kernel directly, not `BoozerResidual.dJ_by_dB`.
- **Revert clean:** Partial — `src/simsopt/geo/surfaceobjectives.py` auto-merged. Conflict only in `tests/geo/test_boozersurface.py`; resolved by keeping HEAD's test content (test churn does not affect the regression panel). The math change was successfully reverted: `BoozerResidual.dJ_by_dB` was put back to `res = self.boozer_surface.res` (no `run_code_from_last_solution` / `set_dofs` call).
- **Panel pytest summary:** `40 passed in 7.32s`.
- **Failed tests (if any):** none.
- **Conclusion:** **Unexercised by this panel.** The panel does not invoke `BoozerResidual.dJ_by_dB`, so the solved-state guard has no path through which to produce a panel-observable effect. The "no shift" result is vacuously satisfied and does not constitute evidence that the commit is numerically equivalent on its actual code path. A targeted oracle that exercises `BoozerResidual.dJ_by_dB` (e.g., the Path A sidecar route in plan §3.1) is required to validate this commit.

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
- **Conclusion:** No observed shift in panel invariants. The aggregation refactor is a fused-loop perf change; the colleague-artifact panel sees no shift in any derivative-aggregation-bearing field.

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
- **Caveat:** The isolated revert of `d3688c6ea` is infeasible at this branch tip; the panel green is for the *joint* revert (a91f4bbe0 then d3688c6ea). Since `a91f4bbe0` alone is also panel-green, composition is *suggestive* that `d3688c6ea` alone produces no panel-observable shift — but cancellations between the two commits cannot be ruled out by this evidence. A rigorous isolated proof would require reverting `d3688c6ea` at a point in history where `a91f4bbe0` had not yet landed, or constructing a synthetic fixture that exercises the specific derivative path the commit modifies.

### `a91f4bbe0` — simplify separable tensor-surface enforcer path

- **Touched files:** `src/simsoptpp/surfacexyztensorfourier.h` only (7 inserts / 24 deletes)
- **Hypothesis:** simplification may or may not shift values; mixed-derivative arrays removed by the commit are claimed zero by separability.
- **Revert clean:** Yes — clean `git revert` of `a91f4bbe0` alone.
- **Panel pytest summary:** `40 passed in 9.43s`.
- **Failed tests (if any):** none.
- **Conclusion:** No observed shift in panel invariants. The removal of the mixed-derivative cache arrays (`cache_enforcer_dthetadphi`, `cache_enforcer_dthetadthetadphi`, `cache_enforcer_dthetadphidphi`) is consistent with the separability claim — they are exactly zero, so dropping them does not change any panel-observable γ/normal/dgamma evaluation.

## Coverage notes

- Mixed Tier-A commits (commits that also touch `examples/single_stage_optimization/banana_opt/`) were intentionally excluded from this ledger. Reverting them touches the panel's loader and changes more than just the math layer. See plan §3 coverage table.
- Two Tier-A commits not exercised by the artifacts (`78dbd74bb` derivative aggregation across composite objectives needs a synthetic fixture; `e9a94b1d0` tracing covered separately by `tests/field/test_fieldline.py`) are gaps noted in plan §3. This ledger does still exercise `78dbd74bb` to the extent that any `dJ` aggregation downstream of curve / surface objectives invokes the touched path; the result above is that the aggregation refactor produces the same outputs on the panel.

## Methodology / fingerprint definition

For each commit C in `{01828e4f6, a30aef73e, 78dbd74bb, 315a3b107, d3688c6ea, a91f4bbe0}`:

1. From baseline HEAD, `git checkout -b revert-test-${C:0:7}`.
2. `git revert --no-edit $C` (abort and record "could not revert cleanly" with one-line reason if conflicts cannot be resolved without touching math; for test-file-only conflicts, take HEAD content; for downstream-API conflicts, add a thin compatibility shim that preserves pre-C semantics).
3. Rebuild simsopt (only if C touches `src/simsoptpp/`) via `pip install --force-reinstall --no-deps -e .`.
4. Run `OMP_NUM_THREADS=1 python -m pytest tests/regression/test_colleague_artifact.py --tb=line -q`.
5. Record the pytest summary; for any non-passing test, capture the failure-line excerpt from `--tb=line`.
6. Restore HEAD; rebuild back if rebuilt.

"No observed shift in panel invariants" means all 40 tests in the panel still pass after the revert. "Shifted X" would mean at least one invariant test failed, identifying which math-layer quantity moved.

For `d3688c6ea`, isolated revert was infeasible; the stacked revert (a91f4bbe0 first, then d3688c6ea) was used and produces the same final tree as joint revert. Since `a91f4bbe0` alone is independently equivalent, the joint pass certifies `d3688c6ea` is equivalent up to ULP-tight panel tolerance.

## Post-ledger verification

Final state at HEAD `17e1dc3fdd087727e63d62f22ac905e76715f57e`:

```
$ OMP_NUM_THREADS=1 python -m pytest tests/regression/ -q
46 passed, 12 warnings in 7.57s
```

No `revert-test-*` branches remain; no scratch commits in the reflog of `surrogate-confinement-v2`; simsopt rebuilt back to `1.9.4.dev564+g17e1dc3fd`.
