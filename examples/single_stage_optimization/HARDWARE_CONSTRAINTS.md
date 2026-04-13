# Constraint Enforcement Summary

## Background

The baseline solver code (`baseline-original`) hardcoded all constraint thresholds as constants. The `candidate-fixed` branch exposed them as CLI arguments to enable automated search. Constraint floors are clamped via `max()` to match baseline defaults.

Updated HBT constraint SSOT:
- TF coil current is fixed at `80 kA`
- banana coil current has an upper limit of `16 kA`
- coil-plasma clearance is `1.5 cm`
- plasma-vessel clearance is `4 cm`
- maximum curvature is `100 m^-1`
- banana winding surface minor radius is `0.21 m`

## Enforced Baseline Floors

All constraint thresholds are clamped via `max()` in the solver code to match the current HBT hardware baseline. CLI arguments below these floors are raised with a printed warning. Optimization weights remain freely adjustable.

### Stage 2 (`banana_coil_solver.py`)

| Constraint | CLI Flag | Baseline Floor | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-threshold` | 0.05m (5cm) | `max(args.cc_threshold, 0.05)` |
| Curvature limit | `--curvature-threshold` | 100 | `max(args.curvature_threshold, 100)` |
| Coil length | `--length-target` | 1.75m | `max(args.length_target, 1.75)` |

### Single-Stage (`single_stage_banana_example.py`)

| Constraint | CLI Flag | Baseline Floor | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-dist` | 0.05m (5cm) | `max(args.cc_dist, 0.05)` |
| Curvature limit | `--curvature-threshold` | 100 | `max(args.curvature_threshold, 100)` |
| Coil-surface clearance | `--cs-dist` | 0.015m (1.5cm) | `max(args.cs_dist, 0.015)` |
| Surface-vessel clearance | `--ss-dist` | 0.04m (4cm) | `max(args.ss_dist, 0.04)` |

**Note:** The current HBT lane fixes the TF current baseline at `80 kA` and uses the tighter coil-plasma clearance plus `100 m^-1` curvature limit as the default hardware contract.

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
