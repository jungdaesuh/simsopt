---
phase: "01"
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified: ["benchmarks/gold_suite/SPEC.md", "src/simsopt/optimize/budget.py", "benchmarks/gold_suite/baselines/desc_alm.json", "benchmarks/gold_suite/baselines/simsopt_alm_2025.json", "benchmarks/gold_suite/oracle.py"]
interactive: false

conventions:
  units: "SI"
  metric: "N/A"
  coordinates: "Cartesian"

dimensional_check:
  quantity_name: "[expected dimension]"

contract:
  scope:
    question: "Which blind matched-budget benchmark problems will constitute the Gold Suite?"
  claims:
    - id: "claim-superiority"
      statement: "The discovered optimizer family beats DESC lsq-auglag and SIMSOPT ALM on blind matched-budget benchmarks."
      deliverables: ["deliv-gold-suite", "deliv-benchmark-report"]
      acceptance_tests: ["test-blind-oracle"]
      references: ["ref-desc-alm", "ref-simsopt-alm"]
  deliverables:
    - id: "deliv-gold-suite"
      kind: "code"
      path: "benchmarks/gold_suite/SPEC.md"
      description: "Gold benchmark suite spanning boundary, equilibrium, and coil optimization"
    - id: "deliv-benchmark-report"
      kind: "report"
      path: "benchmarks/gold_suite/baselines/"
      description: "Comparison report between discovered optimizer and incumbents"
  references:
    - id: "ref-desc-alm"
      kind: "paper"
      locator: "DESC Docs: lsq-auglag"
      role: "benchmark"
      why_it_matters: "Primary documented incumbent"
      applies_to: ["claim-superiority"]
      must_surface: true
      required_actions: ["read", "compare"]
    - id: "ref-simsopt-alm"
      kind: "paper"
      locator: "2025 coil-optimization paper"
      role: "benchmark"
      why_it_matters: "Strongest published ALM workflow in SIMSOPT ecosystem"
      applies_to: ["claim-superiority"]
      must_surface: true
      required_actions: ["use", "compare"]
  acceptance_tests:
    - id: "test-blind-oracle"
      subject: "claim-superiority"
      kind: "oracle"
      procedure: "Evaluate discovered optimizer on Gold Suite with sealed oracle assessment"
      pass_condition: "Statistically significant improvement over incumbents within matched budget"
      evidence_required: ["deliv-benchmark-report"]
  forbidden_proxies:
    - id: "fp-tuning"
      subject: "claim-superiority"
      proxy: "Hyperparameter tuning of existing solvers"
      reason: "Goal is to discover new algorithm families, not tune existing ones"
  links:
    - id: "link-report-superiority"
      source: "claim-superiority"
      target: "deliv-benchmark-report"
      relation: "evaluated_by"
      verified_by: ["test-blind-oracle"]
  uncertainty_markers:
    weakest_anchors: ["Fidelity of SIMSOPT 2025 ALM reproduction"]
    disconfirming_observations: ["Discovery vanishes under Phase 5 ablations"]
---

<objective>
Establish the definitive benchmark suite and incumbent performance contracts. This plan defines the "ground truth" against which all future discovered optimizers will be measured.

Purpose: Locking gold suite and baseline contract
Output: Benchmark specification, budget enforcer, baseline results, and sealed oracle
</objective>

<execution_context>
@/Users/suhjungdae/.codex/get-physics-done/workflows/execute-plan.md
@/Users/suhjungdae/.codex/get-physics-done/templates/summary.md
</execution_context>

<context>
@.gpd/PROJECT.md
@.gpd/ROADMAP.md
@.gpd/STATE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Define Gold Suite problems</name>
  <files>benchmarks/gold_suite/SPEC.md</files>
  <action>Select one fixed-boundary QA case, one free-boundary equilibrium solve, and one single-stage coil optimization case with ReBCO strain constraints. Define evaluation interface in JAX.</action>
  <verify>Check that problems represent boundary, equilibrium, and coil domains.</verify>
  <done>SPEC.md created.</done>
</task>

<task type="auto">
  <name>Task 2: Implement budget enforcement</name>
  <files>src/simsopt/optimize/budget.py</files>
  <action>Create a JAX-traceable operation counter or standardized timer wrapper and BudgetConstraint class.</action>
  <verify>Validate budget enforcement on a simple Scipy optimizer.</verify>
  <done>budget.py implemented.</done>
</task>

<task type="auto">
  <name>Task 3: DESC baseline contract</name>
  <files>benchmarks/gold_suite/baselines/desc_alm.json</files>
  <action>Run DESC lsq-auglag on Gold Suite problems under 3 budget levels. Record convergence and final objectives.</action>
  <verify>Results must match published DESC benchmarks where applicable.</verify>
  <done>desc_alm.json written.</done>
</task>

<task type="auto">
  <name>Task 4: SIMSOPT 2025 ALM baseline</name>
  <files>benchmarks/gold_suite/baselines/simsopt_alm_2025.json</files>
  <action>Reproduce 2025 ALM workflow and run on Gold Suite under 3 budget levels.</action>
  <verify>Results must match or closely approximate the 2025 paper Pareto front.</verify>
  <done>simsopt_alm_2025.json written.</done>
</task>

<task type="auto">
  <name>Task 5: Sealed oracle implementation</name>
  <files>benchmarks/gold_suite/oracle.py</files>
  <action>Create SealedOracle class storing baseline results privately and evaluating candidates.</action>
  <verify>Test oracle with mock results to ensure Victory signal only on statistical improvement.</verify>
  <done>oracle.py implemented.</done>
</task>

</tasks>

<verification>
- Dimensional consistency: All physical metrics (residuals, strain) must use SI units.
- Matched-budget: Budget enforcement must be verified to within 5% tolerance across different hardware.
- Incumbent fidelity: DESC and SIMSOPT baseline runs must match published results for known cases.
</verification>

<success_criteria>
Gold Suite defined, DESC and SIMSOPT baseline contracts locked, Sealed Oracle implemented and ready for blind evaluation.
</success_criteria>

<output>
After completion, create 01-SUMMARY.md.
</output>