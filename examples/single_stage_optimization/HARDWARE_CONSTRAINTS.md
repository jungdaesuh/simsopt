# Hardware Constraint Enforcement Summary

## Background

The baseline solver code (`baseline-original`) hardcoded all constraint thresholds as constants. The `candidate-fixed` branch exposed them as CLI arguments to enable automated search. However, this allowed the search agent to relax constraints below physically buildable limits, producing coils that optimize well on paper but cannot be manufactured.

Per guidance from the hardware team (Rithik Banerjee, 2026-03-17): coil length limits, curvature thresholds, and coil-coil distances are directly linked to hardware feasibility.

## Enforced Hardware Minimums

All constraint thresholds are now clamped via `max()` in the solver code. CLI arguments below these values are raised to the minimum with a printed warning. Optimization weights (cc_weight, curvature_weight, etc.) remain freely adjustable.

### Stage 2 (`banana_coil_solver.py`)

| Constraint | CLI Flag | Hardware Min | Baseline Default | Enforcement |
|-----------|----------|-------------|-----------------|-------------|
| Coil-coil distance | `--cc-threshold` | 0.05m (5cm) | 0.05m | `max(args.cc_threshold, 0.05)` |
| Curvature limit | `--curvature-threshold` | 40 | 40 | `max(args.curvature_threshold, 40)` |
| Coil length | `--length-target` | 1.75m | 1.75m | `max(args.length_target, 1.75)` |

### Single-Stage (`single_stage_banana_example.py`)

| Constraint | CLI Flag | Hardware Min | Baseline Default | Enforcement |
|-----------|----------|-------------|-----------------|-------------|
| Coil-coil distance | `--cc-dist` | 0.05m (5cm) | 0.05m | `max(args.cc_dist, 0.05)` |
| Curvature limit | `--curvature-threshold` | 20 | 20 | `max(args.curvature_threshold, 20)` |
| Coil-surface clearance | `--cs-dist` | 0.02m (2cm) | 0.02m | `max(args.cs_dist, 0.02)` |
| Surface-vessel clearance | `--ss-dist` | 0.04m (4cm) | 0.04m | `max(args.ss_dist, 0.04)` |

## What Is NOT Constrained

Optimization weights control how strongly the solver penalizes constraint violations but do not define the physical limits themselves. These remain freely adjustable:

- `--cc-weight` (coil-coil penalty weight)
- `--curvature-weight` (curvature penalty weight)
- `--length-weight` (coil length penalty weight)
- `--cs-weight` (coil-surface penalty weight)
- `--surf-dist-weight` (surface-vessel penalty weight)
- `--res-weight` (Boozer residual weight)
- `--iotas-weight` (iota tracking weight)
- `--squared-flux-weight` (field error weight, Stage 2 only)

Setting a weight to zero effectively disables the penalty, but the threshold floor still applies to the constraint calculation.

## What Was Wrong Before

The autoresearch agent discovered that relaxing constraints improved field error:

| Parameter | Hardware Min | Agent Used | Improvement |
|-----------|-------------|-----------|-------------|
| `cc_threshold` | 0.05m | 0.021m | FE dropped from 0.012 to 0.004 |
| `curvature_threshold` | 40 (S2 default) | 20 | Prevented self-intersection |

The agent's best Stage 2 result (FE=0.00429) used `cc_threshold=0.021` — a 2.1cm coil spacing that may not be physically achievable. These results need re-evaluation with the hardware minimums enforced.

## Commits

- `275f9181` — Exposed all objective weights as CLI args
