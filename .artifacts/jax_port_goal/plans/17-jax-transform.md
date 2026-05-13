# Item 17 JAX Transform And Memory Plan

## Compiled Boundary

None. `src/simsopt/field/normal_field.py` is byte-identical to upstream
SIMSOPT SHA `1b0cc3a96063197cdbdd01559e04c25456fbe6ff` and contains zero
JAX imports. The only JAX-adjacent code paths reachable from
`NormalField` / `CoilNormalField` are:

- `CoilNormalField.coilset.bs.B()` — the CPU `BiotSavart` parity oracle
  named in the goal-prompt skip list. The C++ kernel runs entirely
  outside JAX.
- `CoilNormalField.optimize_coils` → `coilset.flux_penalty()` →
  `SquaredFlux` — the JAX-native `SquaredFluxJAX` is reachable through
  this path but is owned by item 03 (`SquaredFlux` / `SquaredFluxJAX`)
  and item 04 (`CoilSet` flux penalty). Item 17 does not modify or
  measure this lane.

The Fourier-pair helpers `SurfaceRZFourier.fourier_transform_scalar` and
`SurfaceRZFourier.inverse_fourier_transform_scalar` at
`src/simsopt/geo/surfacerzfourier.py:2169-2323` are pure NumPy with a
double `for` loop over `(m, n)` modes. They do not import JAX. There is
no JAX-native `*_from_spec` / `*_from_dofs` kernel in the JAX core that
corresponds to these helpers, and item 17 does not introduce one — the
SPEC-convention Fourier-pair contract is a thin O(mpol * ntor *
nphi * ntheta) reduction that lives below the production-scale floor
where JIT compile overhead would dominate.

## Transforms

- `jit`: N/A. No JIT entrypoint is touched by item 17.
- `vmap`: N/A.
- `grad`: N/A. `normal_field.py` does not expose a differentiable
  objective; gradients flow through the `Optimizable` / SPEC stack, not
  through this module.
- `scan` / `fori_loop`: N/A.
- `checkpoint` / `remat`: N/A.
- `shard_map` / `pmap` / collectives: N/A.

## Static-Shape Strategy

N/A. No JIT boundary is introduced.

## Memory

The largest array touched by the new closeout test is the production-
scale `BiotSavart.B()` output reshape to `(nphi, ntheta, 3) = (32, 16, 3)`
plus the surface `normal()` array of the same shape, plus a
`(mpol + 1, 2 * ntor + 1)` Vns array at `(5, 7)` or `(4, 5)` depending
on the symmetry branch. Total dense materialization:

- `B[(32, 16, 3)]` — 12288 bytes (float64).
- `surface.normal()[(32, 16, 3)]` — 12288 bytes.
- `bn[(32, 16)]` — 4096 bytes.
- `Vns[(5, 7)]` or `Vns[(4, 5)]` — ≤ 280 bytes.

No buffer donation is used. `donate_argnums` and `donate_argnames` are
N/A: there is no JIT boundary to donate into.

## Dense Materialization Budget

The dense fallback fits in any CPU/GPU ladder budget. The Fourier-pair
helpers are O(mpol * ntor * nphi * ntheta) = O(5 * 7 * 32 * 16) =
17920 multiply-adds per `(m, n)` iteration; the total cost is bounded
by the surface-quadrature grid size. No new dense materialization is
introduced.

## Bench / HLO Artifact

`.artifacts/jax_port_goal/bench/17.json` records that item 17 introduces
no new hot-path kernel; the closeout adds a parity gate only. No timing
benchmark is required. See section 4c carve-out: "no perf change
expected because <one-line justification>" applies because no
implementation change is made; the existing NumPy Fourier helpers and
the CPU `BiotSavart` oracle have full upstream test coverage in
`tests/field/test_normal_field.py` (28 tests) and the C++ Biot-Savart
parity suite under `tests/field/test_biotsavart.py`.

## Sharding / Multi-Device CPU Proxy

N/A. The new test does not introduce any `shard_map`, `psum`,
`all_reduce`, or `pjit` operation. `git grep` over the diff returns zero
hits for those keywords, so the section-4c multi-device CPU proxy
requirement is N/A for item 17.

## CPU-Only Validation Reporting

CPU-only validation is stated explicitly in the commit message and in
the item's `cuda_smoke` field (`not_claimed`). This is a reporting
requirement; section-4c production-scale parity gate is satisfied by
the new closeout test running at `nphi=32`, `ntheta=16`, `ncoils=8`
(4 base coils * 2 nfp expansion via `coils_via_symmetries`).
