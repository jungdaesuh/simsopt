# Item 07 JAX Transform And Memory Plan

## Compiled Boundary

Item 07 introduces no new hot-path kernel; the parent commit already has
the non-distance pure-JAX kernels in place. Each public class composes
the same three transforms:

- `jit` on the scalar `*_pure` kernel (decorated via the module's local
  `jit` helper).
- Reverse-mode `jax.grad` on the kernel with `argnums` chosen to match
  the wrapper's projection contract.
- Numpy host conversion of the gradient blocks before passing them into
  the curve's `d<field>_by_dcoeff_vjp(...)` legacy CPU VJP API.

`FramedCurveTwist` additionally uses `jax.vjp` over `frametwist_pure` to
hand back per-input cotangents, which are then routed through
`rotated_frame_dcoeff_vjp` and `rotated_frame_dash_dcoeff_vjp`.
`LinkingNumber` does not use any JAX transform: `J()` calls the C++
binding and `dJ()` is `Derivative({})` by construction.

## Transform Inventory

| Class | Kernel | jit | grad argnums | vjp | vmap | scan / fori_loop | checkpoint | shard_map / pmap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CurveLength` | `curve_length_pure` | yes (existing) | `(0,)` via `_curve_length_grad` | N/A | N/A | N/A | N/A | N/A |
| `LpCurveCurvature` | `Lp_curvature_pure` | yes | `(0, 1)` via `_lp_curve_curvature_grad` | N/A | N/A | N/A | N/A | N/A |
| `LpCurveCurvatureBarrier` | `curvature_barrier_pure` | yes | `(0, 1)` via `_curvature_barrier_grad` | N/A | N/A | N/A | N/A | N/A |
| `LpCurveTorsion` | `Lp_torsion_pure` | yes | `(0, 1)` via `_lp_curve_torsion_grad` | N/A | N/A | N/A | N/A | N/A |
| `ArclengthVariation` | `curve_arclengthvariation_pure` | yes | `(0,)` via `_curve_arclengthvariation_grad` | N/A | N/A | N/A | N/A | N/A |
| `MeanSquaredCurvature` | `curve_msc_pure` | yes | `(0, 1)` via `_curve_msc_grad` | N/A | N/A | N/A | N/A | N/A |
| `LinkingNumber` | `sopp.compute_linking_number` (C++) | N/A | N/A (zero-derivative contract) | N/A | N/A | N/A | N/A | N/A |
| `FramedCurveTwist (f="lp")` | `frametwist_pure`, `frametwist_lp_pure` | yes | `(0, 1)` via `_frametwist_lp_grad` | yes via `_frametwist_vjp` over `frametwist_pure` | N/A | `lax.fori_loop` inside `frametwist_pure` (existing) | N/A | N/A |
| `FramedCurveTwist (f in {net, range, max})` | `frametwist_pure` + scalar reduction kernel | yes | N/A (`dJ` returns `Derivative({})` by source contract) | N/A | N/A | `lax.fori_loop` inside `frametwist_pure` (existing) | N/A | N/A |

## Static Shape Strategy

All kernels are scalar reductions over per-quadrature-point arrays of
shape `(nquadpoints,)` or `(nquadpoints, 3)`. There is no batching by
ncoils inside the kernels; multi-curve composition happens at the
public wrapper boundary by Python summation of `Derivative`-projected
gradients.

The new closeout tests use:

- `nquadpoints = 32` for the `FramedCurveTwist` `f="lp"` Taylor test
  (small fixture; FD ladder is the dominant work).
- `nquadpoints = 15 * order = 90` for the `LinkingNumber` ncoils=4
  test (default from `create_equally_spaced_curves` at `order=6`).
- `nquadpoints = 32` for the `FramedCurveTwist` `{net, range, max}`
  contract tests.

## Dense Materialization And Donation

The largest array per curve is the `(nquadpoints, 3)` rotated-frame
basis or `(nquadpoints,)` twist profile. Hand-deduced dense memory
budget for the new tests:

- `FramedCurveTwist` lp test: per-curve arrays at
  `(32, 3)` for n1, n2, b1, b2, b1dash, n2dash, plus
  `(32,)` for the twist profile.
- `LinkingNumber` ncoils=4 test: four
  `(90, 3)` `gamma` arrays plus four matching `gammadash` arrays.

No buffer donation (`donate_argnums` / `donate_argnames`) is used. The
JIT-cache hygiene contract (already covered by
`test_framed_curve_twist_reuses_shared_jit_kernels`) ensures the
compiled boundaries remain stable across construction.

## HLO And Bench Artifact

Item 07 introduces no new production hot path. There is no HLO probe
and no timing bench artifact in this item. `.artifacts/jax_port_goal/bench/07.json`
records the rationale (`hot_path_change=false`).

## CUDA Status

CPU-only. JAX transfer-guard documentation notes that fetching CPU
buffers is always allowed, so a transfer-guard-only smoke does not
prove CUDA device residency. No real CUDA artifact is claimed.
