# Constraint Enforcement Summary

## Background

The baseline solver code (`baseline-original`) hardcoded all constraint thresholds as constants. The `candidate-fixed` branch exposed them as CLI arguments to enable automated search. Current HBT contract fields are still configurable where that is useful for searches, but off-contract values fail before solver launch.

Updated HBT constraint SSOT:
- TF coil current is fixed at `80 kA`
- banana coil current has an upper limit of `16 kA`
- coil length uses a `1.9 m` default target with a `2.0 m` hard ceiling
- coil-plasma clearance is `1.5 cm`
- plasma-vessel clearance is `4 cm`
- maximum curvature is `100 m^-1`
- banana winding surface minor radius is `0.21 m`

## Enforced Baseline Limits

All hardware threshold CLI arguments are validated against the current HBT hardware baseline. Values below a floor or above a ceiling raise an error; the solver does not silently clamp them. Optimization weights remain freely adjustable.

### Stage 2 (`banana_coil_solver.py`)

| Constraint | CLI Flag | Baseline Limit | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-threshold` | 0.05m (5cm) | reject below 0.05m |
| Curvature limit | `--curvature-threshold` | 100 | reject above 100m^-1 |
| Coil length target | `--length-target` | 1.9m target, 2.0m hard ceiling | reject above 2.0m |

Stage 2 also enforces the fixed LCFS-to-vessel clearance contract directly on the
loaded plasma boundary. This is not a CLI-tunable floor because the plasma
geometry is inherited from the donor equilibrium, not optimized by Stage 2.
No historical off-spec bypass exists for this clearance gate.

### Single-Stage (`single_stage_banana_example.py`)

| Constraint | CLI Flag | Baseline Limit | Enforcement |
|-----------|----------|---------------|-------------|
| Coil-coil distance | `--cc-dist` | 0.05m (5cm) | reject below 0.05m |
| Curvature limit | `--curvature-threshold` | 100 | reject above 100m^-1 |
| Coil length target | `--length-target` | 1.9m target, 2.0m hard ceiling | reject above 2.0m |
| Coil-surface clearance | `--cs-dist` | 0.015m (1.5cm) | reject below 0.015m |
| Surface-vessel clearance | `--ss-dist` | 0.04m (4cm) | reject below 0.04m |

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

Setting a weight to zero disables that soft penalty term, but hardware gates and final artifact checks still enforce the contract.

## ALM Contract

ALM uses one scalar penalty schedule for all constraints. `constraint_blocks`
are diagnostic labels for summaries, history, and benchmark grouping; they do
not define solver-update groups or independent penalty schedules.

Legacy result schemas may still contain `alm_block_penalties=None`. That value
is intentional compatibility metadata for readers that know the historical
field; it is not a missing calculation.

Every ALM evaluator must emit the normalized arrays consumed by the solver:
`constraint_values`, `feasibility_values`, and `dual_update_values`. Stage-2
signal routing also requires the explicit hard, surrogate, and hard-dual arrays
at the call sites that consume those signals. All ALM signal arrays must match
`constraint_values.shape`; a shape mismatch is a contract error.

ALM checkpoint resume is strict. Checkpoints must carry `constraint_names` in
the exact current order, along with `penalty` and `multipliers`. Older
checkpoints without `constraint_names` are unsafe for multiplier reuse and must
not be resumed as if they matched the current constraint schema.
