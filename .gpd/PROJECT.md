# Meta-Agent Optimizer Discovery for Constrained Stellarators

## What This Is

This research project aims to discover entirely new families of constrained stellarator optimizers using a self-referential, metacognitive algorithm-discovery system. Built on top of SkyDiscover and HyperAgents, the system co-evolves solver code and search policies, validated against blind matched-budget benchmarks spanning boundary, equilibrium, and coil optimization.

## Core Research Question

Can a self-referential, metacognitive algorithm-discovery system co-evolve solver code and search policy to discover new families of constrained stellarator optimizers that outperform current incumbents (DESC `lsq-auglag` and SIMSOPT ALM) on blind matched-budget benchmarks?

## Scoping Contract Summary

### Contract Coverage

- **Claim / deliverable:** Discovered optimizer family beats DESC `lsq-auglag` and SIMSOPT ALM on blind matched-budget benchmarks.
- **Acceptance signal:** Statistically significant improvement over incumbents within matched budget on Gold Suite with sealed oracle assessment.
- **False progress to reject:** Hyperparameter tuning of existing solvers.

### User Guidance To Preserve

- **User-stated observables:** Matched-budget convergence (objective vs budget), Final equilibrium residual.
- **User-stated deliverables:** Gold benchmark suite, Comparison report with ablation results.
- **Must-have references / prior outputs:** DESC `lsq-auglag` docs, 2025 SIMSOPT coil-optimization paper, `gpu_jax_pro.md`, `jax_single_stage_parity_audit_2026-03-29.md`.
- **Stop / rethink conditions:** Discovered algorithm disappears under specified ablations (no self-improving meta-agent, no open-ended archive, etc.).

### Scope Boundaries

**In scope**

- Locking gold suite and baseline contract
- Discovery of new optimizer families via SkyDiscover/HyperAgents substrate
- Co-evolution of search strategy (EvoX-style)
- Distillation and simplification of top winners
- Rigorous ablation studies

**Out of scope**

- Generic hyperparameter tuning of existing ALM
- Framework-specific solver improvements only

### Active Anchor Registry

- **ref-desc-alm**: DESC Docs: lsq-auglag
  - Why it matters: Primary documented incumbent
  - Carry forward: planning | execution | verification
  - Required action: read | compare
- **ref-simsopt-alm**: 2025 coil-optimization paper
  - Why it matters: Strongest published ALM workflow in SIMSOPT ecosystem
  - Carry forward: planning | execution | verification
  - Required action: use | compare

### Carry-Forward Inputs

- `gpu_jax_pro.md`
- `jax_single_stage_parity_audit_2026-03-29.md`
- `simsopt-jax` current best single-stage optimization

### Skeptical Review

- **Weakest anchor:** Fidelity of SIMSOPT 2025 ALM reproduction
- **Unvalidated assumptions:** Co-evolved search strategy generalizes to unseen problems
- **Competing explanation:** Observed wins are due to better hyperparameter defaults rather than structural discovery
- **Disconfirming observation:** Discovery vanishes under Phase 5 ablations
- **False progress to reject:** Hyperparameter tuning of existing solvers

### Open Contract Questions

- Which blind matched-budget benchmark problems will constitute the Gold Suite?
- How to faithfully reproduce the 2025 SIMSOPT coil-optimization ALM workflow?

## Physics Subfield

Computational Plasma Physics — Stellarator Optimization

## Mathematical Framework

Augmented Lagrangian Methods (ALM), Co-evolutionary Algorithm Discovery, Implicit Differentiation (IFT).

## Notation Conventions

To be established during initial phases.

## Unit System

SI units (m, T, A, etc.)

## Computational Tools

- JAX / JAXLIB (for autodiff and GPU acceleration)
- SkyDiscover (discovery loop)
- HyperAgents (meta-procedure)
- EvoX (co-evolution)
- SIMSOPT-JAX (target framework)
- DESC (incumbent framework)

## Requirements

### Validated

(None yet — derive and validate to confirm)

### Active

- [ ] **REQ-01**: Lock the gold suite and baseline contract (Phase 1)
- [ ] **REQ-02**: Run discovery islands (exploit, explore, weird) (Phase 2)
- [ ] **REQ-03**: Turn on co-evolution of search policy (Phase 3)
- [ ] **REQ-04**: Distill and simplify winning optimizers (Phase 4)
- [ ] **REQ-05**: Perform rigorous ablation studies (Phase 5)

### Out of Scope

(To be refined as project progresses)

## Key References

- **DESC Docs**: lsq-auglag (Primary documented incumbent)
- **2025 Coil-Optimization Paper**: Strongest published SIMSOPT-side ALM workflow

## Target Publication

High-impact physics or machine learning journal (e.g., Nature Machine Intelligence, Journal of Computational Physics).

## Constraints

- **matched-budget**: Comparisons must be performed under matched computational budget (wall-clock or traceable ops).
- **blind-oracle**: Evaluation must be performed against a sealed oracle to ensure objective assessment.

## Key Decisions

| Decision | Rationale | Outcome |
| --- | --- | --- |
| Minimal initialization — defer deep scoping | Fast project bootstrap | — Pending |

---

_Last updated: 2026-04-06 after initialization (minimal)_
