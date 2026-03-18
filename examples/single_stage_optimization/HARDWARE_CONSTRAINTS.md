# Constraint Enforcement Summary

## Background

The baseline solver code (`baseline-original`) hardcoded all constraint thresholds as constants. The `candidate-fixed` branch exposed them as CLI arguments to enable automated search. Constraint floors are clamped via `max()` to match baseline defaults.

Per Rithik Banerjee (2026-03-17): coil length limits, curvature thresholds, and coil-coil distances are the parameters most directly linked to hardware feasibility. Current values are **working defaults** that may change as the design evolves.

## Enforced Baseline Floors

All constraint thresholds are clamped via `max()` in the solver code to match baseline-original values. CLI arguments below these floors are raised with a printed warning. Optimization weights remain freely adjustable.

### Stage 2 (`banana_coil_solver.py`)

| Constraint | CLI Flag | Baseline Floor | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-threshold` | 0.05m (5cm) | `max(args.cc_threshold, 0.05)` |
| Curvature limit | `--curvature-threshold` | 20 (baseline default: 40) | `max(args.curvature_threshold, 20)` |
| Coil length | `--length-target` | 1.75m | `max(args.length_target, 1.75)` |

### Single-Stage (`single_stage_banana_example.py`)

| Constraint | CLI Flag | Baseline Floor | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-dist` | 0.05m (5cm) | `max(args.cc_dist, 0.05)` |
| Curvature limit | `--curvature-threshold` | 20 | `max(args.curvature_threshold, 20)` |
| Coil-surface clearance | `--cs-dist` | 0.02m (2cm) | `max(args.cs_dist, 0.02)` |
| Surface-vessel clearance | `--ss-dist` | 0.04m (4cm) | `max(args.ss_dist, 0.04)` |

**Note:** Both solvers enforce a floor of 20. The Stage 2 baseline default was 40 and the single-stage baseline default was 20. The agent should explore both CT=20 and CT=40 for Stage 2.

## What Is NOT Constrained

Optimization weights control how strongly the solver penalizes constraint violations. These remain freely adjustable:

- `--cc-weight` (coil-coil penalty weight)
- `--curvature-weight` (curvature penalty weight)
- `--length-weight` (coil length penalty weight)
- `--cs-weight` (coil-surface penalty weight)
- `--surf-dist-weight` (surface-vessel penalty weight)
- `--res-weight` (Boozer residual weight)
- `--iotas-weight` (iota tracking weight)
- `--squared-flux-weight` (field error weight, Stage 2 only)

Setting a weight to zero effectively disables the penalty, but the threshold floor still applies to the constraint calculation.
