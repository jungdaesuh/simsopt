# Research Requirements: Meta-Agent Optimizer Discovery

**Defined:** 2026-04-06
**Core Research Question:** Can a self-referential, metacognitive algorithm-discovery system co-evolve solver code and search policy to discover new families of constrained stellarator optimizers that outperform current incumbents?

## Primary Requirements

### Benchmarking (Gold Suite)

- [ ] **GOLD-01**: Define blind matched-budget benchmark problems spanning boundary, equilibrium, and coil optimization.
- [ ] **GOLD-02**: Establish baseline performance contract for DESC `lsq-auglag`.
- [ ] **GOLD-03**: Establish baseline performance contract for SIMSOPT 2025 ALM workflow.
- [ ] **GOLD-04**: Implement a sealed oracle for objective assessment of discovered optimizers.

### Discovery (Phase 2 & 3)

- [ ] **DISC-01**: Set up SkyDiscover/HyperAgents substrate for solver code co-evolution.
- [ ] **DISC-02**: Run exploit, explore, and "weird" discovery islands for solver structural rewrites.
- [ ] **DISC-03**: Implement EvoX-style co-evolution of the search policy.

### Distillation and Ablation (Phase 4 & 5)

- [ ] **ABLA-01**: Distill top winning optimizers into simplest functional variants.
- [ ] **ABLA-02**: Perform 8+ rigorous ablation checks (no meta-agent, no co-evolved policy, etc.).
- [ ] **ABLA-03**: Verify that discovery "win" persists under all specified ablations.

## Follow-up Requirements

### Extended Analysis

- **EXTD-01**: Generalize discovered optimizers to non-stellarator constrained optimization problems.
- **EXTD-02**: Integrate discovered search policies into standard library optimizers.

## Out of Scope

| Topic | Reason |
| --- | --- |
| Hyperparameter tuning only | Violates core discovery goal |
| Manual heuristic design | Replaced by co-evolutionary system |

## Accuracy and Validation Criteria

| Requirement | Accuracy Target | Validation Method |
| --- | --- | --- |
| GOLD-01 | Representative physics problems | Peer review of suite selection |
| GOLD-02/03 | Reproduced faithfully | Comparison with official documentation/code |
| DISC-02 | Novel solver structures | Code audit of evolved artifacts |
| ABLA-03 | Statistical significance | Blind matched-budget evaluation |

## Contract Coverage

| Requirement | Decisive Output / Deliverable | Anchor / Benchmark / Reference | Prior Inputs / Baselines | False Progress To Reject |
| --- | --- | --- | --- | --- |
| GOLD-01 | deliv-gold-suite | ref-desc-alm, ref-simsopt-alm | simsopt-jax best | toy problems only |
| DISC-02 | deliv-optimizer-code | SkyDiscover | HyperAgents | param-tuning only |
| ABLA-03 | deliv-benchmark-report | ref-desc-alm, ref-simsopt-alm | Baseline contract | qualitative win only |

## Traceability

| Requirement | Phase | Status |
| --- | --- | --- |
| GOLD-01 | Phase 1: Baseline & Gold Suite | Pending |
| GOLD-02 | Phase 1: Baseline & Gold Suite | Pending |
| GOLD-03 | Phase 1: Baseline & Gold Suite | Pending |
| GOLD-04 | Phase 1: Baseline & Gold Suite | Pending |
| DISC-01 | Phase 2: Discovery | Pending |
| DISC-02 | Phase 2: Discovery | Pending |
| DISC-03 | Phase 3: Co-evolution | Pending |
| ABLA-01 | Phase 4: Distillation | Pending |
| ABLA-02 | Phase 5: Ablation | Pending |
| ABLA-03 | Phase 5: Ablation | Pending |

**Coverage:**

- Primary requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---

_Requirements defined: 2026-04-06_
