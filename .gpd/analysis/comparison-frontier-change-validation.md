---
date: 2026-04-13
theory_source: working-tree frontier/comparison/preserved-timeout change set in examples/single_stage_optimization and its regression tests
data_source: local observational evidence from pytest runs and dry-run command executions on the same workspace
overall_agreement: good
chi2_ndof: 0.0
p_value: 1.0
max_tension_sigma: 0.0
comparison_verdicts:
  - subject_id: frontier_regression_slice
    subject_kind: artifact
    subject_role: decisive
    reference_id: pytest-frontier-slice-2026-04-13
    comparison_kind: baseline
    metric: relative_error
    threshold: "0 unexpected failures"
    verdict: pass
    recommended_action: "Keep this slice as the fast regression gate for frontier/comparison work."
  - subject_id: full_geo_regression
    subject_kind: artifact
    subject_role: decisive
    reference_id: pytest-geo-full-2026-04-13
    comparison_kind: baseline
    metric: relative_error
    threshold: "0 unexpected failures"
    verdict: pass
    recommended_action: "Treat full tests/geo as the upstream/downstream confirmation gate."
  - subject_id: comparison_wrapper_init_only_guard
    subject_kind: artifact
    subject_role: decisive
    reference_id: dry-run-init-only-seed-guard-2026-04-13
    comparison_kind: baseline
    metric: relative_error
    threshold: "exact contract match"
    verdict: pass
    recommended_action: "Keep the init-only guard enabled by default."
  - subject_id: frontier_metadata_contract
    subject_kind: artifact
    subject_role: decisive
    reference_id: targeted-frontier-metadata-tests-2026-04-13
    comparison_kind: baseline
    metric: relative_error
    threshold: "exact contract match"
    verdict: pass
    recommended_action: "Use preserved-timeout payloads and comparison summaries as stable downstream consumers."
---

# Experimental Comparison: Frontier Change Validation

**Date:** 2026-04-13
**Theory source:** the intended behavior of the current change set:

- `single_stage_banana_example.py` should implement `frontier_tradeoff_score_v1`, emit frontier metadata, and preserve target-mode behavior.
- `single_stage_objectives.py` should support normalized frontier objective terms without regressing target-mode callers.
- `run_single_stage_goal_mode_comparison.py` should reject init-only Stage 2 seeds unless explicitly overridden and should record comparison metadata consistently.

**Observed data sources:** repository-local execution artifacts, treated here as the observational side because no external experimental dataset was specified.

This is therefore an **implementation-vs-observation** comparison, not a theory-vs-laboratory-data comparison.

## Data Source Metadata

| Source | Experiment/Observatory | Year | Observable | Conditions | Systematic Uncertainties | Reference |
|--------|------------------------|------|------------|------------|--------------------------|-----------|
| `pytest-frontier-slice` | local `pytest` | 2026 | touched frontier/comparison/ALM/basin-hopping suites | Python 3.13.12, current working tree | deterministic software test oracle | `308 passed in 7.55s` |
| `pytest-geo-full` | local `pytest` | 2026 | full `tests/geo` upstream/downstream sweep | same workspace | deterministic software test oracle; warnings only | `535 passed, 4 skipped, 11 warnings in 1363.45s` |
| `dry-run-no-allow` | local CLI execution | 2026 | init-only Stage 2 seed rejection | same workspace, `--dry-run`, no override | deterministic exit contract | exit code `1`, explicit `--allow-init-only-stage2-seed` guidance |
| `dry-run-allow` | local CLI execution | 2026 | init-only Stage 2 seed override and summary recording | same workspace, `--dry-run --allow-init-only-stage2-seed` | deterministic dry-run summary content | exit code `0`, `stage2_artifact_init_only=true` |
| `targeted-frontier-tests` | local `pytest` | 2026 | frontier metadata payload and comparison summary contract | same workspace | deterministic software test oracle | `1 passed` and `3 passed` focused checks |

### Data Quality Notes

- All observations were taken from the same local workspace and interpreter family used for the implementation work.
- `pytest-geo-full` includes warnings from permanent-magnet-grid tests, but no failures.
- `dry-run-allow` validates wrapper semantics only. It does **not** provide a new matched non-dry-run physics result.

## Unit Conversion Checklist

This comparison is on software-contract observables rather than physical units.

- [x] Count units: pass/fail/skip counts compared directly; no conversion needed.
- [x] Exit-code units: compared as exact integers; no conversion needed.
- [x] Boolean/JSON contract fields: compared by exact match; no conversion needed.
- [x] Natural-unit restoration: not applicable.
- [x] Detector acceptance / efficiency corrections: not applicable.

## Convention Matching

| Convention | Theory | Observation | Conversion needed? |
|------------|--------|-------------|--------------------|
| Goal-mode forwarding | wrapper must pass `--single-stage-goal-mode` explicitly for both lanes | observed in dry-run summary command arrays | no |
| Init-only seed admissibility | wrapper must reject `init_only=true` seeds unless override is present | observed as exit `1` without override and exit `0` with override | no |
| Frontier payload metadata | frontier payloads/summaries must record implementation tag and frontier metadata | observed via focused passing tests and summary expectations | no |
| Regression contract | touched suites and full `tests/geo` must show zero unexpected failures | observed exactly | no |

## Comparison Table

For deterministic software checks, the "pull" is reported as `0` on exact match and would be `1` on mismatch. It is a contract-check surrogate, not a statistical uncertainty estimate.

| Quantity | Theory | Experiment | Exp. Uncertainty | Theory Uncertainty | Pull (sigma) | Status |
|----------|--------|------------|------------------|-------------------|--------------|--------|
| Frontier regression slice failures | `0` unexpected failures | `0` failures, `308` passed | exact-match | exact-match | `0` | agree |
| Full `tests/geo` failures | `0` unexpected failures | `0` failures, `535` passed, `4` skipped | exact-match | exact-match | `0` | agree |
| Init-only seed guard without override | nonzero exit with explicit override guidance | exit `1`; raised `ValueError` naming `--allow-init-only-stage2-seed` | exact-match | exact-match | `0` | agree |
| Init-only seed override path | zero exit and summary records `stage2_artifact_init_only=true` | exit `0`; dry-run summary records `stage2_artifact_init_only=true` and both goal modes | exact-match | exact-match | `0` | agree |
| Frontier metadata contract | frontier payload/reporting exposes implementation tag and frontier metadata | focused tests passed for `SINGLE_STAGE_GOAL_MODE_IMPL`, `BOOZER_SURFACE_TARGET_VOLUMES`, and `FRONTIER_TRUST_OK` | exact-match | exact-match | `0` | agree |

## Statistical Analysis

### Global Fit Quality

This is a deterministic contract comparison, not a noisy measurement comparison. To keep the report quantitative, use the exact-match surrogate:

- **Chi-squared:** `0.0`
- **Degrees of freedom:** `4`
- **Chi-squared / DOF:** `0.0`
- **p-value:** `1.0`
- **Interpretation:** perfect agreement between predicted contract behavior and observed workspace evidence

### Individual Pulls

| Observable | Pull (sigma) | Direction | Systematic dominated? |
|-----------|--------------|-----------|-----------------------|
| Frontier regression slice | `0` | none | yes; deterministic |
| Full `tests/geo` regression | `0` | none | yes; deterministic |
| Init-only guard | `0` | none | yes; deterministic |
| Frontier metadata payload | `0` | none | yes; deterministic |

### Correlations

- `pytest-geo-full` is a superset confirmation and is therefore correlated with the touched regression slice.
- The focused frontier metadata tests overlap with the larger `308`-test slice; they are kept here because they directly observe the highest-signal new contract fields.

## Systematic Corrections

| Correction | Magnitude | Applied to | Method | Reference |
|-----------|-----------|------------|--------|-----------|
| Workspace/environment alignment | fixed | observation | same repo and Python environment for all commands | local workspace |
| Dry-run scope caveat | qualitative | interpretation | explicitly mark dry-run evidence as wrapper-semantics-only | local analysis |

## Discrepancy Classification

No discrepancies were observed in the implementation-contract comparison.

### Residual Analysis

- No failing regression tests were observed in the touched slice or the full `tests/geo` sweep.
- The init-only seed guard behaved exactly as designed in both the rejecting and overridden paths.
- No downstream consumer mismatch was observed in frontier payload/reporting tests.

## Important Boundary

This report does **not** establish that frontier mode is already better in physics terms than target mode. It only establishes that the implemented contract matches the observed workspace behavior.

The missing observational dataset is:

- a matched, non-dry-run `target` vs `frontier` comparison from the same **feasible, non-init-only** seed

That runtime experiment is still needed to answer:

- whether frontier improves iota/volume without unacceptable QA or Boozer trust degradation
- whether the new scalarization is actually preferable in practice

## Figures

- `artifacts/comparisons/frontier-change-validation/observations.json`: machine-readable observation dataset for this comparison
- `artifacts/comparisons/frontier-change-validation/plot_change_validation.py`: helper script to render a compact bar chart / status figure from the observation dataset

## Summary

**Overall agreement:** Excellent at the implementation-contract level.

**Key tensions:** None in the observed software behavior.

**Missing data:** A matched non-dry-run target-vs-frontier experiment on a feasible, non-init-only seed.

**Recommended actions:**

1. Run the matched target-vs-frontier experiment from the shortened feasible bridge checkpoint.
2. Compare feasible iota, volume, QA, Boozer trust/rejects, and hardware metrics.
3. If frontier wins on the feasible frontier, keep the mode; if not, retune frontier weights or trust threshold.
