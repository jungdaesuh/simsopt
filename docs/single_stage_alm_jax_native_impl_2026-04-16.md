# Single-Stage ALM JAX-Native Status

Date: 2026-04-24

## Goal

Keep the ALM JAX/ondevice product lane native on CPU/GPU and remove the
shared target-lane host bridge from `examples/single_stage_optimization/alm_utils.py`
without regressing the preserved CPU/reference SciPy lane.

## Current Status

- [x] Single-stage ALM resolves and passes `inner_optimizer_contract` on the
  JAX/ondevice lane.
- [x] Single-stage ALM builds and uses the native traceable ALM runtime bundle.
- [x] Shared target-lane ALM inner solves now require a native
  `target_inner_value_and_grad`.
- [x] The old target-lane `jax.pure_callback` fallback is removed from
  `alm_utils.py`.
- [x] Stage 2 ALM on the JAX/ondevice lane now supplies a native ALM
  `value_and_grad` built from the Stage 2 target objective bundle.
- [x] CPU/reference ALM still uses the host/SciPy lane unchanged.
- [x] Single-stage native ALM JAXPR coverage remains callback-free.
- [x] Stage 2 native ALM now has callback-free target-objective coverage.

## Shared Contract

The shared ALM seam is now strict:

- `minimize_alm(..., inner_optimizer_contract=TargetOptimizerContract(...))`
  must also receive `target_inner_value_and_grad=...`.
- Target-lane ALM inner solves no longer synthesize a host bridge through
  `jax.pure_callback`.
- Reference/SciPy ALM remains available when `inner_optimizer_contract is None`.

This is the intended contract because the JAX/ondevice lane is the product
surface. A target-lane request must execute with native JAX objective and
gradient evaluation, not a host callback fallback.

## Live Call Paths

Single-stage:

- `resolve_single_stage_alm_inner_optimizer_contract(...)` in
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- traceable ALM runtime bundle builder in the same file
- `minimize_alm(..., target_inner_value_and_grad=alm_target_value_and_grad, ...)`
  in the ALM branch

Shared seam:

- strict target-lane contract in
  `examples/single_stage_optimization/alm_utils.py`

Stage 2:

- native Stage 2 target objective bundle in
  `src/simsopt/objectives/stage2_target_objective_jax.py`
- Stage 2 ALM ondevice wiring in
  `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`

## Validation Targets

- [x] Single-stage ALM integration tests prove the native lane does not need
  SciPy and now rejects missing native target evaluators.
- [x] Single-stage compiled ALM runtime JAXPR contains no `pure_callback`.
- [x] Stage 2 target-objective JAXPR contains no `pure_callback`.
- [x] Stage 2 native ALM value/gradient matches the host ALM evaluation on the
  same state.

## Deferred Items

- [ ] Real CUDA signoff on actual hardware

`Issue 38` is no longer deferred here. It was root-fixed at the winding-surface
radius contract and is tracked as fixed in
`examples/single_stage_optimization/ISSUES.md`.

## Notes

- The old plan item “single-stage ALM does not pass `inner_optimizer_contract`”
  is no longer true.
- The old plan item “the native ALM lane still depends on `jax.pure_callback`”
  is no longer true for the product JAX/ondevice paths.
- The remaining launch-scope work after this change is validation/signoff
  work, not more fallback cleanup on the single-stage ALM lane.
