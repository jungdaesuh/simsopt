# Plan: Phase 1 — Baseline & Gold Suite

## Overview
Establish the definitive benchmark suite and incumbent performance contracts. This plan defines the "ground truth" against which all future discovered optimizers will be measured.

## Contract
- **Claims:** [claim-superiority]
- **Deliverables:** [deliv-gold-suite, deliv-benchmark-report]
- **References:** [ref-desc-alm, ref-simsopt-alm]
- **Acceptance Tests:** [test-blind-oracle]
- **Forbidden Proxies:** [fp-tuning]
- **Uncertainty Markers:** ["Fidelity of SIMSOPT 2025 ALM reproduction"]

## Tasks

<task id="01-01" wave="1">
<context>
Define the specific physics problems for the Gold Suite.
</context>
<do>
1. Select one fixed-boundary QA optimization case (e.g., Ginsburg-15).
2. Select one free-boundary equilibrium solve case.
3. Select one single-stage coil optimization case with ReBCO strain constraints.
4. Define the evaluation interface for these problems in JAX.
</do>
<done>
Gold Suite problem specification document created in `benchmarks/gold_suite/SPEC.md`.
</done>
</task>

<task id="01-02" wave="1">
<context>
Implement the matched-budget enforcement infrastructure.
</context>
<do>
1. Create a JAX-traceable operation counter or standardized timer wrapper.
2. Define the `BudgetConstraint` class to interrupt optimization when the budget is exceeded.
3. Validate budget enforcement on a simple Scipy optimizer.
</do>
<done>
Budget enforcement code implemented in `src/simsopt/optimize/budget.py`.
</done>
</task>

<task id="01-03" wave="2" depends_on="01-01, 01-02">
<context>
Establish the DESC `lsq-auglag` baseline contract.
</context>
<do>
1. Integrate DESC `lsq-auglag` into the Gold Suite interface.
2. Run DESC on the Gold Suite problems under 3 different budget levels (Low, Med, High).
3. Record convergence curves and final objective values.
4. Lock the baseline results in `benchmarks/gold_suite/baselines/desc_alm.json`.
</do>
<done>
DESC baseline contract locked.
</done>
</task>

<task id="01-04" wave="2" depends_on="01-01, 01-02">
<context>
Establish the SIMSOPT 2025 ALM baseline contract.
</context>
<do>
1. Reproduce the 2025 ALM workflow using SIMSOPT-JAX components.
2. Run the reproduced workflow on the Gold Suite problems under the same 3 budget levels.
3. Record convergence curves and final objective values.
4. Lock the baseline results in `benchmarks/gold_suite/baselines/simsopt_alm_2025.json`.
</do>
<done>
SIMSOPT 2025 baseline contract locked.
</done>
</task>

<task id="01-05" wave="3" depends_on="01-03, 01-04">
<context>
Implement the sealed oracle.
</context>
<do>
1. Create a `SealedOracle` class that stores baseline results privately.
2. Implement the `Oracle.evaluate(candidate_results)` method.
3. Ensure the oracle returns a "Victory" signal only for statistically significant improvements over ALL incumbents.
</do>
<done>
Sealed oracle implemented in `benchmarks/gold_suite/oracle.py`.
</done>
</task>

## Verification
- **Dimensional consistency:** All physical metrics (residuals, strain) must use SI units.
- **Matched-budget:** Budget enforcement must be verified to within 5% tolerance across different hardware.
- **Incumbent fidelity:** DESC and SIMSOPT baseline runs must match published results for known cases.
