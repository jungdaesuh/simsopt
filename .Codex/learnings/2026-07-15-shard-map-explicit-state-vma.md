---
date: 2026-07-15
problem: Boozer trajectory sharding passed single-device tests but failed on a forced two-device mesh through captured state and VMA branch mismatches.
tags: [jax, shard-map, sharding, testing]
---

# Pass device state explicitly and preserve varyingness across shard-map branches

## Problem

The Boozer batched tracing path used `shard_map`, but its field state and event
arrays were captured lexically. They stayed committed to the default device, so
the second mesh device could not use them. After that placement bug was fixed,
strict varying-manual-axes checking exposed a second issue: an invariant
`zeros((4,))` conditional branch was paired with a branch derived from the
varying trajectory state.

## Dead ends

- A test hard-coded `jax.devices("cpu")`; under `JAX_PLATFORMS=cuda` it failed
  before exercising production code. Device discovery had to follow the active
  backend.
- Wrapping the entire multi-device test in `jax.transfer_guard("disallow")`
  rejected JAX 0.10's internal scalar index staging, not a public-path transfer.
  The narrower existing transfer test plus explicit output-device assertions
  were the discriminating checks.
- Merely closing over frozen field arrays did not replicate them. Lexical capture
  is not an input-placement contract.
- A shape- and dtype-compatible constant zero branch still failed because VMA
  varyingness is part of the branch type.

## Working approach

1. Reproduce with two forced CPU devices so `check_vma=True` is exercised even
   on a one-GPU allocation.
2. Place every array leaf of the shardable field/spec/event PyTree with replicated
   `NamedSharding` and pass that tree as an explicit `shard_map` argument with
   matching `P()` input specs.
3. Keep opaque level-set callbacks on the unsharded path because their closures
   can own device arrays but are not replicable PyTrees.
4. Replace the invariant event branch with `zeros_like(y0)` so both `lax.cond`
   branches inherit the trajectory operand's varyingness.
5. Assert numerical parity and actual two-device output placement, then repeat
   the regression on two A100 GPUs.

## Why it worked

`shard_map` reasons about both physical placement and whether values vary along
manual mesh axes. Explicit PyTree inputs make placement inspectable and enforceable;
deriving neutral branch values from the sharded operand preserves the VMA type.
Opaque callable closure state has neither property, so representation-aware
fallback is safer than pretending it can be replicated.

## Reusable rule

When a `shard_map(check_vma=True)` body touches an array, require that array to be
an explicit input with a declared sharding spec; for every conditional, construct
neutral outputs from the corresponding sharded operand, and keep non-PyTree
callbacks unsharded unless their state is made explicit.

## Pointers

- `src/simsopt_jax/core/sharding.py:238`
- `src/simsopt_jax/core/tracing.py:3202`
- `src/simsopt_jax/core/tracing.py:3638`
- `tests/jax/core/test_tracing_jax_item14.py:1103`
- Commit `ef4c86815`
- Perlmutter jobs `55946255` and `55946576`
