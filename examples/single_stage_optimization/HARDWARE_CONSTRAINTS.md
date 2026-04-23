# Constraint Enforcement Summary

## Background

The baseline solver code (`baseline-original`) hardcoded all constraint thresholds as constants. The `candidate-fixed` branch exposed them as CLI arguments to enable automated search. Lower-bound floors are clamped via `max()`, while upper-bound ceilings are clamped via `min()`, to match the HBT hardware contract.

Updated HBT constraint SSOT:
- TF coil current is fixed at `80 kA`
- banana coil current has an upper limit of `16 kA`
- coil length uses a `1.9 m` default target with a `2.0 m` hard ceiling
- coil-plasma clearance is `1.5 cm`
- plasma-vessel clearance is `4 cm`
- maximum curvature is `100 m^-1`
- banana winding surface minor radius is `0.21 m`

## Enforced Baseline Limits

All constraint thresholds are clamped in the solver code to match the current HBT hardware baseline. CLI arguments below a floor are raised with a printed warning, while values above a ceiling are lowered with a printed warning. Optimization weights remain freely adjustable.

### Stage 2 (`banana_coil_solver.py`)

| Constraint | CLI Flag | Baseline Limit | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-threshold` | 0.05m (5cm) | `max(args.cc_threshold, 0.05)` |
| Curvature limit | `--curvature-threshold` | 100 | `min(args.curvature_threshold, 100)` |
| Coil length target | `--length-target` | 1.9m target, 2.0m hard ceiling | `min(args.length_target, 2.0)` |

Stage 2 also enforces the fixed LCFS-to-vessel clearance contract directly on the
loaded plasma boundary. This is not a CLI-tunable floor because the plasma
geometry is inherited from the donor equilibrium, not optimized by Stage 2.
Historical off-spec reproduction can bypass the check only via
`ACCEPT_OFFSPEC_PLASMA_VESSEL_CLEARANCE=1`.

### Single-Stage (`single_stage_banana_example.py`)

| Constraint | CLI Flag | Baseline Limit | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-dist` | 0.05m (5cm) | `max(args.cc_dist, 0.05)` |
| Curvature limit | `--curvature-threshold` | 100 | `min(args.curvature_threshold, 100)` |
| Coil length target | `--length-target` | 1.9m target, 2.0m hard ceiling | `min(args.length_target, 2.0)` |
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
