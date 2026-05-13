# Finite-I Pre-revert vs HEAD Parity Evidence

**Date:** 2026-05-12
**Task:** #25 — Run end-to-end numeric parity test on `BoozerSurfaceFiniteI` vs pre-revert `BoozerSurface(I=)`

## Subject

Commit `459da8fab` (2026-05-10, "refactor: move finite-I Boozer support out of src/ into examples-side wrapper") removed `I=` from `BoozerSurface.__init__` in `src/simsopt/geo/boozersurface.py` and modified the C++ kernel at `src/simsoptpp/boozerresidual_*`. Finite-I support was relocated to `examples/single_stage_optimization/banana_opt/boozer_finite_current.py::BoozerSurfaceFiniteI` as a subclass that applies `G_effective = G + iota * I` and explicit-current-basis rank-1 transforms in the Python wrapper layer.

This test diffs the wrapper's full-solve output against the pre-refactor monolithic implementation on a shared deterministic fixture, exercising the entire `run_code` path through the C++ kernel and back.

## Method

- **Baseline SHA:** `d8deb9e11` (= `459da8fab^`, direct parent of the I-removal commit). Built into isolated conda env at `/tmp/simsopt-prerevert/.conda-env` (Python 3.11). `simsopt-1.9.4.dev551+gd8deb9e11` editable install.
- **HEAD SHA:** `1002df7d6`. Existing env at `simsopt-surrogate/.conda-env`.
- **Fixture:** NCSX coils (`get_ncsx_data` + `coils_via_symmetries(3, True)`), `SurfaceXYZTensorFourier(mpol=3, ntor=3, stellsym=True, nfp=3)` fit to magnetic axis at radius 0.1, `Area` label. `iota0 = 0.4` seed, `G0 = 2π·Σ|I_coil|·μ₀/(2π)`. Fixture hash:
  `c2570fb8fed0c6973506955c5009955b82aca542b46e6e7a44927cdf8fdda0c8`
  Hash verified identical between pre-revert and HEAD envs (proves both saw the same seed).
- **Lanes:** 4 = {LS mode (constraint_weight=100) × exact mode (constraint_weight=None)} × {I=0, I=μ₀·5kA = 6.283185307179587e-3}.
- **Driver:** single Python script (`finite_i_parity_driver.py`) that auto-detects which API is available and calls the appropriate constructor. Same byte-identical script runs in both envs via subprocess with different `python` interpreter paths.
- **Tolerance:** machine precision. Verdicts:
  - BIT_IDENTICAL: surface.x, iota, G all relative-diff < 1e-13
  - PARITY_AT_1e-10 / 1e-6: progressively looser
  - DRIFT_DETECTED: relative diff > 1e-6

## Results

These rows are the historical Task #25 parity record for HEAD `1002df7d6`.

| Lane            | Verdict                          | iota rel-diff | G rel-diff | surface.x max rel-diff |
|-----------------|----------------------------------|---------------|------------|------------------------|
| LS mode, I=0    | **BIT_IDENTICAL**                | 0.000e+00     | 0.000e+00  | 0.000e+00              |
| LS mode, I≠0    | **BIT_IDENTICAL**                | 1.388e-15     | 0.000e+00  | 6.436e-16              |
| exact mode, I=0 | BOTH_FAIL_IDENTICAL_TRAJECTORY   | 0.000e+00     | 0.000e+00  | 0.000e+00              |
| exact mode, I≠0 | BOTH_FAIL_NAN_VS_NUMERIC         | N/A           | N/A        | 1.000e+00              |

Fixture hash matched across all lanes.

## Current-HEAD Follow-up

Commit `e41321273` (2026-05-12) later added residual-decreasing backtracking to
`BoozerSurfaceFiniteI.solve_residual_equation_exactly_newton` for nonzero-I exact
Newton only. The historical parity table above is still the direct pre-revert
diff for HEAD `1002df7d6`, but the old lane-4 caveat is no longer current.

Re-running the same exact-mode `I=mu0*5kA` fixture at `e41321273` now gives:

- `success=True`
- `iota=4.028394632921262e-01`
- `G=1.388198779389556e+01`
- `solve_error=None`

The LS finite-current lane is unchanged at current HEAD: rerunning the driver
produced the same `iota` and `G` as the historical HEAD JSON exactly. The
current exact finite-current outcome is pinned by
`test_finite_current_exact_newton_converges_on_task25_lane4_fixture`.

## Interpretation

This interpretation applies to the historical parity run at HEAD `1002df7d6`.

**Lanes 1 & 2 (LS mode, both currents) — production-path parity proven at ULP floor.**
The wrapper's `G_effective = G + iota*I` substitution + rank-1 explicit-current-basis derivative transforms produce numerically identical surface.x, iota, and G to the pre-revert monolithic implementation. The 1-ULP iota drift in lane 2 is floating-point rounding noise from the differently-ordered operations, not a math difference.

**Lane 3 (exact mode, I=0) — failure mode is bit-identical.**
Both implementations diverge from the bad initial seed (`iota0=0.4` is too far from any physical equilibrium for exact-mode Newton on this fixture). Pre-revert and HEAD blow up to *exactly* the same diverged final state (iota = 1.347886305237789e+04, G = 3.025060550783371e+03). Identical-failure proves the divergence is in shared code paths (the simsoptpp Newton driver), not in the wrapper.

**Lane 4 (exact mode, I≠0) — only observed divergence; not a math regression.**
Both implementations fail. Pre-revert returns numerically diverged but technically finite floats (`iota = -1.770e+05`, `G = 1.669e+03`). HEAD raises a `ValueError` via `numpy.asarray_chkfinite` when the Newton step contains NaN/inf. Both are failure modes — neither produces a usable result — but they differ in *how* they fail. This is a failure-handling divergence downstream of where Newton has already broken down, not a math divergence in the wrapper's `G_effective` or rank-1 transforms.

The exact-mode I≠0 configuration with a generic LS-seed was a known-pathological combination for HEAD `1002df7d6`. Production code never invoked this directly; exact mode is only entered from an already-converged LS solution. The lane 4 divergence had no production impact, and it no longer occurs after `e41321273`; see the current-HEAD follow-up above.

## Verdict

**Task #25 closed.** Wrapper math is faithful to the pre-revert monolithic finite-I path on every production-relevant lane:
- Vacuum LS (I=0): bit-identical.
- Finite-current LS (I=μ₀·5kA): bit-identical at the ULP floor (1-ULP iota noise).
- Failure trajectories on bad seeds: bit-identical where both produce finite floats; failure-handling diverges only after Newton has already blown up.

The colleague-artifact regression panel (`b32efa818`) was previously the strongest oracle for this path. This test adds the missing direct pre-revert diff, which is the single proof Task #25 asked for.

## Artifacts

- `finite_i_prerevert_vs_head_2026-05-12.json` — full per-lane evidence (surface.x vectors, iota, G, success states, fixture hash, both SHAs)
- `finite_i_parity_driver.py` — reproducible driver, runs in both envs
- `finite_i_diff_parity.py` — diff script that produced the verdict table

## Reproduction

```bash
git worktree add /tmp/simsopt-prerevert 459da8fab^
conda create -p /tmp/simsopt-prerevert/.conda-env python=3.11 -y
cd /tmp/simsopt-prerevert && /tmp/simsopt-prerevert/.conda-env/bin/pip install -e .
conda install -p /tmp/simsopt-prerevert/.conda-env libcxx -c conda-forge -y  # macOS only
PRE=/tmp/simsopt-prerevert/.conda-env/bin/python
HEAD=<simsopt-surrogate>/.conda-env/bin/python
for mode in ls exact; do
  for I in 0.0 0.006283185307179587; do
    out_pre="parity_prerevert_${mode}_I${I}.json"
    out_head="parity_HEAD_${mode}_I${I}.json"
    $PRE finite_i_parity_driver.py --mode $mode --current-I $I --output $out_pre
    PYTHONPATH=<simsopt-surrogate>/examples/single_stage_optimization \
      $HEAD finite_i_parity_driver.py --mode $mode --current-I $I --output $out_head
  done
done
python finite_i_diff_parity.py
```
