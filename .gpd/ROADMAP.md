# Roadmap: Meta-Agent Optimizer Discovery

## Overview

This project follows a structured discovery loop to evolve new constrained stellarator optimizers. We begin by locking a "Gold Suite" of benchmarks and establishing baseline contracts for DESC and SIMSOPT incumbents. We then run evolutionary discovery islands, turn on co-evolution of search policies, and finally distill and rigorously ablate the results to ensure a genuine scientific discovery.

## Phases

- [ ] **Phase 1: Baseline & Gold Suite** - Lock benchmarks and incumbent performance contracts.
- [ ] **Phase 2: Discovery** - Run evolutionary islands for solver code mutation.
- [ ] **Phase 3: Co-evolution** - Evolve search policy alongside solver code.
- [ ] **Phase 4: Distillation** - Simplify and clean top-performing optimizers.
- [ ] **Phase 5: Ablation** - Verify discovery against 8+ ablation checks.

## Phase Details

### Phase 1: Baseline & Gold Suite

**Goal:** Establish the ground truth and matched-budget constraints for the discovery system.
**Depends on:** Nothing
**Requirements:** [GOLD-01, GOLD-02, GOLD-03, GOLD-04]
**Success Criteria:**
1. Gold Suite defined with boundary, equilibrium, and coil problems.
2. DESC `lsq-auglag` performance contract locked.
3. SIMSOPT 2025 ALM workflow performance contract locked.
4. Sealed oracle implemented and ready for blind evaluation.

### Phase 2: Discovery

**Goal:** Evolve novel solver architectures via code mutation on multiple islands.
**Depends on:** Phase 1
**Requirements:** [DISC-01, DISC-02]
**Success Criteria:**
1. SkyDiscover/HyperAgents loop active.
2. Exploit, explore, and "weird" islands produce structural solver rewrites.

### Phase 3: Co-evolution

**Goal:** Evolve the search strategy itself using EvoX-style co-evolution.
**Depends on:** Phase 2
**Requirements:** [DISC-03]
**Success Criteria:**
1. Search policy evolves to adaptively allocate budget and switch mutation strategies.

### Phase 4: Distillation

**Goal:** Distill complex winners into minimal functional implementations.
**Depends on:** Phase 3
**Requirements:** [ABLA-01]
**Success Criteria:**
1. Simplest variant that maintains the "win" is identified and documented.

### Phase 5: Ablation

**Goal:** Rigorously test the discovery against 8+ ablation scenarios.
**Depends on:** Phase 4
**Requirements:** [ABLA-02, ABLA-03]
**Success Criteria:**
1. Discovery persists under all Specified ablations (e.g., no meta-agent, no open-ended archive).

## Progress

| Phase | Plans Complete | Status | Completed |
| --- | --- | --- | --- |
| 1. Baseline & Gold Suite | 0/TBD | Not started | - |
| 2. Discovery | 0/TBD | Not started | - |
| 3. Co-evolution | 0/TBD | Not started | - |
| 4. Distillation | 0/TBD | Not started | - |
| 5. Ablation | 0/TBD | Not started | - |
