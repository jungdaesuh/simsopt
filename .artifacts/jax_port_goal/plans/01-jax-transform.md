# Item 01 JAX Transform And Memory Plan

## CurrentPenalty

Compiled boundary:

- `current_penalty_pure(I, threshold)` is a scalar pure function.
- The hot path is scalar and shape-stable: `I` and `threshold` are scalar
  float values or scalar JAX arrays.

Transforms:

- `jit`: supported and tested on device-resident scalar input.
- `vmap`: supported and tested on device-resident vector input.
- reverse-mode derivative: the legacy wrapper uses `strict_scalar_grad`, which
  builds the scalar cotangent from the primal value and avoids the raw
  `jax.grad` 0-D derivative-block failure.
- `scan`, `fori_loop`, `checkpoint`, `shard_map`, `pmap`: N/A. The kernel is a
  scalar elementwise penalty and has no loop or partitioned array body.

Memory and donation:

- Largest array in the direct item 01 test is the two-entry `vmap` input
  `float64[2]`; scalar wrapper value/gradient paths operate on scalar leaves.
- No buffer donation is used. `donate_argnums` and `donate_argnames` are N/A
  because wrappers may reuse `Current` state after value/gradient evaluation.

## Distance Wrappers

Compiled boundaries:

- Candidate cullers in `_distance_jax.py` pad point clouds to static pair mask
  shapes, compute JAX masks, then hostify candidate tuples for legacy public
  wrapper semantics.
- `cc_distance_pure`, `cc_distance_barrier_pure`, and `cs_distance_pure`
  consume explicit geometry arrays and threshold scalars.
- Shared `_pairwise_reductions.py` helper boundaries consume explicit point
  arrays and optional chunk-size arguments.

Transforms:

- `jit`: candidate mask helpers and gradient wrappers are jitted by existing
  decorators/tests.
- `vmap`: not required for public distance-wrapper reductions; row/block work
  is expressed with `lax.scan`.
- `scan`: chunked pairwise reductions use nested `lax.scan` blocks to bound
  dense materialization.
- `checkpoint` / `remat`: inner chunk scans use `jax.checkpoint` for block
  recomputation rather than retaining all intermediate pairwise matrices.
- sharding: shared `_pairwise_reductions.py` rowwise helpers use
  `maybe_shard_pairwise_row_inputs`; Stage 2 target paths use
  `maybe_shard_pairwise_row_trees`. CPU proxy evidence exists, but no CUDA
  sharding/performance claim is made here.
- `pmap` / collectives: N/A for item 01 public wrappers.

Memory and donation:

- Dense path largest temporary is `(row_count, col_count)` pairwise distances.
- Chunked path largest temporary is bounded by
  `(chunk_size, chunk_size)` pairwise distances plus row masks and weights.
- Default pairwise chunk sizes are supplied by backend runtime policy.
- No buffer donation is used. Public wrappers and tests may reuse geometry
  arrays after objective evaluation.

HLO / benchmark artifact:

- `.artifacts/jax_port_goal/bench/01.json` records that item 01 is not a new
  production hot path and cites focused transfer/chunking validation instead of
  a timing benchmark.
