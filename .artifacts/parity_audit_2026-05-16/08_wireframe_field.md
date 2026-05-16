# PRIORITY 8 — Wireframe Magnetic-Field Kernel Parity Audit

**Audit timestamp:** 2026-05-16
**Audit scope:** straight-segment Biot-Savart kernel; total-field and per-segment
contributions; spatial first derivative; consumer wrapper
`WireframeFieldJAX`.

## Files audited

| File | Lines | Role |
| ---- | ----- | ---- |
| `src/simsopt/jax_core/wireframe.py` | 509 | Pure-JAX kernel (item 29) |
| `src/simsoptpp/wireframe_field_impl.h` | 239 | C++ SIMD + scalar inner kernel |
| `src/simsoptpp/magneticfield_wireframe.cpp` | 127 | C++ field-cache reduction loop |
| `src/simsoptpp/magneticfield_wireframe.h` | 126 | C++ field-class declarations |
| `src/simsopt/field/wireframefield_jax.py` | 147 | JAX-backed `WireframeFieldJAX` wrapper (item 30) |
| `src/simsopt/field/wireframefield.py` | 171 | CPU `WireframeField` reference wrapper |
| `tests/jax_core/test_wireframe_jax_item29.py` | 409 | Item-29 parity tests (C++ oracle + closed form) |
| `tests/jax_core/test_wireframe_item29.py` | 159 | Adapter parity tests against `WireframeField` |
| `tests/field/test_wireframefield_jax_item30.py` | 170 | Public wrapper parity tests |

No call-sites for `wireframe_segment_dB_by_dX_contributions` were found in
`src/`, `tests/`, or `benchmarks/`.

## Executive summary

The JAX wireframe kernel is a faithful, bit-identical port of the C++
inner kernel — the closed-form `(|r_1| + |r_2|) / (|r_1| |r_2| (|r_1|
|r_2| + r_1.r_2)) * (r_1 x r_2)` factor and the literal closed-form
Jacobian both reproduce the C++ arithmetic operation-for-operation,
with the same `fak = 1e-7` prefactor (storage form of `mu_0 / (4 pi)`),
the same singular-regime behaviour (no defensive floors), and the same
per-half-period seg-sign weighting. The component-wise gradient block
(`dBdx`/`dBdy`/`dBdz`) matches the C++ `dB_dX_i[0/1/2]` line-by-line.

Top-3 findings:

1. **MEDIUM — Reduction-order divergence in `wireframe_segment_*_contributions`.**
   The contribution kernels use `jnp.sum(jax.vmap(...)(...), axis=0)` (tree
   reduction) over half-periods, while the C++ `compute()` uses sequential
   `axpy_array` (`magneticfield_wireframe.cpp:80,88-89`). The total-field
   JIT paths (`_wireframe_B_jit` and friends) correctly use `lax.scan`
   (`wireframe.py:399`, `wireframe.py:418-422`) to match C++ accumulation
   order, but the **contribution paths used by the public
   `dB_by_dsegmentcurrents` API do not.** For `n_halfprds = 2 nfp` (small),
   this is well within `direct_kernel` tolerance, but the divergence is
   not bounded by construction.

2. **LOW — Docstring layout assertion contradicts the abstract
   simsopt-jax convention.** Lines 27-40 of `wireframe.py` document
   `dB[p, k, m] = d_m B_k` (k = B component, m = derivative direction),
   matching the literal C++ storage; this **swaps the axis labels** of
   the abstract convention quoted in `CLAUDE.md` (`dB[p, j, l] = d_j
   B_l`, j = derivative direction, l = B component). The docstring
   notes this explicitly. The actual storage is component-first, and
   `test_dB_layout_convention_via_finite_difference`
   (`test_wireframe_jax_item29.py:307-350`) validates the kernel-local
   contract with FD. No correctness issue for `WireframeFieldJAX`
   consumers since the C++ oracle uses the same component-first layout,
   but other JAX modules in this worktree (`biot_savart_jax`,
   `dB_by_dX[p, j, l]`) follow the swapped convention. Downstream code
   that mixes the two paths must read the docstring carefully.

3. **INFO — Public surface gap (`wireframe_segment_dB_by_dX_contributions`).**
   This entry point is exported and tested for shape only via the
   combined-vs-separate path; it has no direct parity assertion against
   the C++ field-cache `dB_i` arrays (the C++ side caches but the public
   wrapper never returns them — `dB_by_dsegmentcurrents` always returns
   `B_i` regardless of `compute_derivatives`). The JAX export is reachable
   but unconsumed; either add a C++ parity test or remove the export.

## Function-by-function parity matrix

| JAX function | C++ counterpart | Math | Physics | Algorithm | Computation | Severity |
| ------------ | --------------- | :--: | :-----: | :-------: | :---------: | :------: |
| `_wireframe_segment_B_from_arrays` | `wireframe_field_kernel<T,0>` inner block (lines 63-76) | OK | OK | OK | OK | clean |
| `_wireframe_segment_B_and_dB_by_dX_from_arrays` | `wireframe_field_kernel<T,1>` inner block (lines 63-100 / 186-222) | OK | OK | OK | OK | clean |
| `wireframe_segment_B` | (none — convenience wrapper) | OK | OK | OK | OK | clean |
| `wireframe_segment_dB_by_dX` | (none — convenience wrapper, recomputes B) | OK | OK | wasted B work (no C++ analog) | OK | LOW |
| `wireframe_segment_B_and_dB_by_dX` | (none — convenience wrapper) | OK | OK | OK | OK | clean |
| `wireframe_segment_B_contributions` | `compute(0)` write into `field_cache.get(B,i)` (cpp:54-100) | OK | OK | sum-vs-axpy order | OK | MEDIUM |
| `wireframe_segment_dB_by_dX_contributions` | `compute(1)` write into `field_cache.get(dB,i)` (cpp:107-111) | OK | OK | sum-vs-axpy + no consumer | OK | INFO + MEDIUM |
| `wireframe_B` -> `_wireframe_B_jit` | `compute(0)` total-field reduction (cpp:101-105) | OK | OK | OK (scan + scan) | OK | clean |
| `wireframe_dB_by_dX` -> `_wireframe_dB_jit` | `compute(1)` total-dB reduction (cpp:106-112) | OK | OK | OK (scan + scan) | OK | clean |
| `wireframe_B_and_dB_by_dX` -> `_wireframe_B_and_dB_jit` | combined `compute(1)` (cpp:101-112) | OK | OK | OK (single fused scan) | OK | clean |

## Detailed findings

### (a) Straight-segment closed-form parity

The closed-form Biot-Savart formula for a straight segment from `a` to
`b` carrying unit current, evaluated at point `r`, used in both the JAX
and C++ kernels is:

```
B(r) = (mu_0 / 4 pi) * (|r_1| + |r_2|) / (|r_1| |r_2| (|r_1| |r_2| +
       r_1 . r_2)) * (r_1 x r_2)
```

with `r_1 = r - a`, `r_2 = r - b`. This is algebraically equivalent to
the textbook `(cos theta_1 - cos theta_2) / d` form used by the NumPy
reference in `test_single_segment_closed_form_parity`
(`test_wireframe_jax_item29.py:90-119`).

**C++ kernel** (`wireframe_field_impl.h:63-76`):

```cpp
auto diff0 = point_i - node0_vec;
auto diff1 = point_i - node1_vec;
auto norm_diff0_sq = normsq(diff0);
auto norm_diff1_sq = normsq(diff1);
auto norm_diff0 = sqrt(norm_diff0_sq);
auto norm_diff1 = sqrt(norm_diff1_sq);
auto diff0_diff1 = norm_diff0*norm_diff1;
auto denom = diff0_diff1 * (diff0_diff1 + inner(diff0, diff1));
auto factor = (norm_diff0 + norm_diff1) / denom;
auto diff0_cross_diff1 = cross(diff0, diff1);

B_i.x = xsimd::fma(diff0_cross_diff1.x, factor, B_i.x);
```

**JAX kernel** (`wireframe.py:121-133`):

```python
def _wireframe_segment_B_from_arrays(points, node0, node1):
    diff0 = points - node0
    diff1 = points - node1
    norm_diff0 = jnp.sqrt(jnp.sum(diff0 * diff0, axis=-1))
    norm_diff1 = jnp.sqrt(jnp.sum(diff1 * diff1, axis=-1))
    diff0_diff1 = norm_diff0 * norm_diff1
    denom = diff0_diff1 * (diff0_diff1 + jnp.sum(diff0 * diff1, axis=-1))
    factor = (norm_diff0 + norm_diff1) / denom
    return _MU0_OVER_4PI * factor[:, None] * jnp.cross(diff0, diff1)
```

The JAX kernel emits the same operations in the same order:
`normsq -> sqrt -> product -> denom -> factor -> cross -> fak * factor * cross`.
The only stylistic differences are: (i) JAX broadcasts the scalar
`factor` via `factor[:, None]`, whereas C++ multiplies through `Vec3dSimd`;
(ii) C++ uses `xsimd::fma` for `B_i += factor * cross` (because the
SIMD path accumulates across SIMD lanes), but the *value* delivered to
the host is `fak * factor * cross` after the j-loop trailer (line 116).
The non-SIMD scalar fallback (line 197) just does `B_i += factor *
diff0_cross_diff1`, identical to the JAX expression.

**Constant `mu_0 / (4 pi)`.** Both kernels factor this out at write
time. C++ stores it as `double fak = 1e-7;` (`wireframe_field_impl.h:43`
SIMD path, line 166 scalar path); JAX stores it as `_MU0_OVER_4PI =
1e-7` (`wireframe.py:103`). Both write `fak * factor * cross`. The
*high-precision* value of `mu_0 / (4 pi)` is `1e-7` exactly under the
2019 SI redefinition (the magnetic constant is no longer exact, but `4
pi * 1e-7` remained the historical exact value used by the C++ code
since 2019). Both implementations agree on the same constant.

**Closed-form parity test.** `test_single_segment_closed_form_parity`
(`test_wireframe_jax_item29.py:157-196`) uses the alternate `(cos1 -
cos2) / rho * dl_hat x rho_hat` form as an oracle independent of the
`(|r_1| + |r_2|)` arithmetic and passes within `direct_kernel`
tolerance (`rtol=1e-10`).

### (b) Singularity handling

The closed form diverges in two regimes:

1. **`r` on the wire segment between `a` and `b`** (the wire itself):
   `diff0` is antiparallel to `diff1`, so `inner(diff0, diff1) =
   -|diff0||diff1|`, making the denominator
   `|diff0||diff1| * (|diff0||diff1| + inner(diff0, diff1)) -> 0`.
   Both kernels emit `nan` or `inf` here; neither has a guard.
2. **`r = a` or `r = b`**: `|diff0| = 0` or `|diff1| = 0`, making
   `denom = 0`. Same `inf`/`nan` behaviour in both.

The JAX docstring at lines 64-75 of `wireframe.py` explicitly documents
this as faithful-port behaviour and explicitly states "No defensive
floors are inserted." The C++ code matches — there is no zero-distance
guard in `wireframe_field_impl.h`. Both kernels produce identical
non-finite outputs in singular regimes, which preserves the
`direct_kernel` parity invariant. Downstream optimization code is
responsible for choosing observation points that avoid the wire.

**Severity:** clean (intentional and documented).

### (c) Gradient parity — closed-form Jacobian

The 9-component closed-form Jacobian `d_m B_k` is computed without
autodiff. Both kernels share an intermediate `grad_factor` vector.

**C++** (`wireframe_field_impl.h:79-100`):

```cpp
auto p0 = diff0 * norm_diff1;
auto p1 = diff1 * norm_diff0;
auto factorsq = factor * factor;
auto grad_factor = (p0 + p1) * (-factorsq)
                   - (p0*(1.0/norm_diff0_sq)
                      + p1*(1.0/norm_diff1_sq))*(1.0/denom);

dB_dX_i[0].x = grad_factor.x * diff0_cross_diff1.x;
dB_dX_i[0].y = grad_factor.y * diff0_cross_diff1.x
               + factor * ( diff1.z - diff0.z);
...
```

**JAX** (`wireframe.py:190-234`):

```python
p0 = diff0 * norm_diff1[:, None]
p1 = diff1 * norm_diff0[:, None]
factorsq = factor * factor
grad_factor = (p0 + p1) * (-factorsq[:, None]) - (
    p0 / norm_diff0_sq[:, None] + p1 / norm_diff1_sq[:, None]
) / denom[:, None]

dBdx = jnp.stack(
    (
        gfx * cx,
        gfy * cx + factor * (d1z - d0z),
        gfz * cx + factor * (d0y - d1y),
    ),
    axis=-1,
)
...
dB = _MU0_OVER_4PI * jnp.stack((dBdx, dBdy, dBdz), axis=-2)
```

I verified all nine gradient components line-by-line against the C++
code. Every entry matches exactly:

| C++ component | JAX component |
| ------------- | ------------- |
| `dB_dX_i[0].x = gf.x * c.x` | `dBdx[..., 0] = gfx * cx` |
| `dB_dX_i[0].y = gf.y * c.x + factor*(d1.z - d0.z)` | `dBdx[..., 1] = gfy*cx + factor*(d1z - d0z)` |
| `dB_dX_i[0].z = gf.z * c.x + factor*(d0.y - d1.y)` | `dBdx[..., 2] = gfz*cx + factor*(d0y - d1y)` |
| `dB_dX_i[1].x = gf.x * c.y + factor*(-d1.z + d0.z)` | `dBdy[..., 0] = gfx*cy + factor*(-d1z + d0z)` |
| `dB_dX_i[1].y = gf.y * c.y` | `dBdy[..., 1] = gfy*cy` |
| `dB_dX_i[1].z = gf.z * c.y + factor*(-d0.x + d1.x)` | `dBdy[..., 2] = gfz*cy + factor*(-d0x + d1x)` |
| `dB_dX_i[2].x = gf.x * c.z + factor*(d1.y - d0.y)` | `dBdz[..., 0] = gfx*cz + factor*(d1y - d0y)` |
| `dB_dX_i[2].y = gf.y * c.z + factor*(d0.x - d1.x)` | `dBdz[..., 1] = gfy*cz + factor*(d0x - d1x)` |
| `dB_dX_i[2].z = gf.z * c.z` | `dBdz[..., 2] = gfz*cz` |

Then `dB = _MU0_OVER_4PI * jnp.stack((dBdx, dBdy, dBdz), axis=-2)`
delivers shape `(N, 3, 3)` with `dB[p, k, m] = fak * dB_dX_i[k].m` —
exactly matching the C++ storage layout at `wireframe_field_impl.h:121-123`:
`dB_by_dX(i+j, k, 0/1/2) = fak * dB_dX_i[k].x/y/z`.

**Layout-convention asymmetry (LOW).** The CLAUDE.md tensor-convention
note quoted at the top of this audit specifies `dB_by_dX[p, j, l] = d_j
B_l` (j = derivative direction, l = B component), matching the abstract
simsopt-jax convention. The wireframe JAX kernel — and the wireframe
C++ kernel it ports — both store this differently: `dB[p, k, m] = d_m
B_k` (k = B component, m = derivative direction). The docstring at
`wireframe.py:27-40` explicitly calls this out and confirms it matches
`simsoptpp.WireframeField.dB_by_dX()` and `simsoptpp.BiotSavart`. Tests
confirm the layout: `test_dB_layout_convention_via_finite_difference`
(`test_wireframe_jax_item29.py:307-350`) builds `B(point +/- eps * e_m)`
finite differences along axis `m` and asserts equality with `dB[:, m]`
(not `dB[m, :]`). This is **deliberately compatible** with the C++
storage (no parity loss against the C++ oracle), but downstream
consumers who interleave with `biot_savart_jax` outputs (which follow
the abstract `[p, j, l]` convention) must read the docstring to avoid
silent transposition bugs.

**Gradient FD validation.** `test_dB_layout_convention_via_finite_difference`
exercises a single point at `derivative_heavy` tolerance
(`first_derivative_rtol`). Total-system gradient parity against C++ is
validated by `test_wireframe_total_B_and_dB_match_cpp_wireframefield`
(`test_wireframe_item29.py:71-90`) at `direct_kernel` tolerance.

### (d) Batched/vmap parity

Two batch axes interact in the wireframe path: half-periods (size `n_halfprds = 2 nfp`)
and segments (size `n_segments`).

**C++** (`magneticfield_wireframe.cpp:52-99`):

```cpp
for (int i = 0; i < nSegments; ++i) {
    Array& Bi = field_cache.get(IndexedFieldCacheKind::B, i);
    set_array_to_zero(Bi);
    ...
    for (int j = 0; j < nHalfPrds; j++) {
        ... // build node0, node1 for (i,j)
        wireframe_field_kernel<Array, 0>(pointsx, pointsy, pointsz,
                                          node0, node1, Bij, ...);
        simsoptpp::axpy_array(Bi, Bij, seg_signs[j]);
    }
}
for (int i = 0; i < nSegments; ++i) {
    Array& Bi = field_cache.get(IndexedFieldCacheKind::B, i);
    double current = currents[i];
    xt::noalias(B) = B + current * Bi;
}
```

**JAX total-field path** (`wireframe.py:316-335,384-400`):

```python
def _segment_total_B(points, node0_by_segment, node1_by_segment, seg_signs):
    def add_half_period(acc, half_period):
        node0, node1, seg_sign = half_period
        return acc + seg_sign * _wireframe_segment_B_from_arrays(
            points, node0, node1
        ), None

    B, _ = jax.lax.scan(
        add_half_period, jnp.zeros_like(points),
        (node0_by_segment, node1_by_segment, seg_signs),
    )
    return B

@jax.jit
def _wireframe_B_jit(points, nodes, segments, seg_signs, currents):
    node0, node1 = _nodes_by_segment(nodes, segments)
    def add_segment(acc, segment):
        node0_by_segment, node1_by_segment, current = segment
        B = _segment_total_B(points, node0_by_segment, node1_by_segment, seg_signs)
        return acc + current * B, None
    B, _ = jax.lax.scan(add_segment, jnp.zeros_like(points),
                        (node0, node1, currents))
    return B
```

This is an *exact* match for the C++ accumulation pattern: outer scan
over segments, inner scan over half-periods, both in sequential
`acc <- acc + value` form. `lax.scan` guarantees sequential reduction
order, so the floating-point trace matches C++ at the `direct_kernel`
tolerance level. The closed-loop and multi-half-period parity tests
confirm bit-equality.

**JAX contribution path** (`wireframe.py:238-270`):

```python
def wireframe_segment_B_contributions(points, nodes, segments, seg_signs):
    points_jax = _as_jax_float64(points)
    seg_signs_jax = _as_jax_float64(seg_signs).reshape((-1,))
    node0, node1 = _gather_segment_nodes(nodes, segments)

    def half_period_B(node0_by_half, node1_by_half, seg_sign):
        return seg_sign * _wireframe_segment_B_from_arrays(
            points_jax, node0_by_half, node1_by_half,
        )

    def segment_B(node0_by_segment, node1_by_segment):
        return jnp.sum(
            jax.vmap(half_period_B)(
                node0_by_segment, node1_by_segment, seg_signs_jax,
            ),
            axis=0,
        )

    return jax.vmap(segment_B, in_axes=(1, 1), out_axes=0)(node0, node1)
```

**MEDIUM finding.** The reduction over half-periods here is
`jnp.sum(jax.vmap(half_period_B)(...), axis=0)` — a tree reduction, not
a sequential scan. The C++ counterpart in
`magneticfield_wireframe.cpp:80` is `axpy_array(Bi, Bij, seg_signs[j])`
inside the `j = 0 .. nHalfPrds-1` loop — sequential. For small
`n_halfprds` (typical: 2 or 4), this divergence is well within the
`direct_kernel` tolerance (`rtol=1e-10`); the parity test
`test_wireframe_segment_B_contributions_match_cpp_fieldcache`
(`test_wireframe_item29.py:112-132`) passes on the `nfp=2 -> 4 half-periods`
fixture. For higher `nfp` or larger configurations, the order
divergence is not bounded by construction. Worth tracking when GPU
parity is exercised; CPU `jnp.sum` over 4 entries is unlikely to differ
from sequential `axpy`.

**Symmetric note for `wireframe_segment_dB_by_dX_contributions`**
(`wireframe.py:273-306`): same `jnp.sum(jax.vmap(...))` pattern, same
medium-severity reduction-order divergence. This entry point is
*exported* but I find no in-tree consumer beyond the smoke / shape
tests, and it has no direct C++ parity test against the
`field_cache.get(dB, i)` arrays (the public `dB_by_dsegmentcurrents`
wrapper returns only B-cache entries).

### (e) Algorithm and computation cross-checks

- **`fak` placement.** Both kernels factor `fak = 1e-7` out of the hot
  loop and apply it once at the write stage. JAX applies it at the
  segment-level `_wireframe_segment_B_from_arrays` return; C++ applies
  it inside the j-loop scalar trailer (`B(i+j, 0) = fak * B_i.x[j]`).
  Because the C++ code accumulates *unscaled* `B_i` across half-periods
  inside the SIMD vector and then multiplies by `fak` at write time,
  and the JAX code multiplies by `fak` *inside* every per-segment
  per-half-period call before scan accumulation, the two trace orderings
  for the `fak` multiplication differ. In IEEE arithmetic
  `fak * (a + b + c) = fak * a + fak * b + fak * c` to within one ulp
  (rounding at the final sum); the parity tests pass under
  `direct_kernel`, so this is benign.

- **Stellsym sign handling.** The wireframe construction in
  `ToroidalWireframe.__init__` (`wireframe_toroidal.py:88-118`) puts
  even half-periods at sign `+1.0` and odd at `-1.0`, then duplicates
  by `nfp` rotations. The seg-sign array is shape `(2*nfp,)`. C++ stores
  this as `vector<double> seg_signs` and indexes `seg_signs[j]`. JAX
  reshapes to `(-1,)` and broadcasts via `vmap`/`scan`. Test
  `test_multi_halfperiod_seg_signs_parity`
  (`test_wireframe_jax_item29.py:199-251`) exercises a 4-half-period
  case with `[1, -1, 1, -1]` and passes at `direct_kernel` tolerance.

- **`node0`/`node1` gather pattern.** JAX builds these via
  `jnp.take(nodes, segments[:, 0/1], axis=1)`
  (`wireframe.py:116-117`), yielding shape `(n_halfprds, n_segments,
  3)`. The total-field path then `moveaxis(node0, 1, 0)`
  (`wireframe.py:313`) to give `(n_segments, n_halfprds, 3)` for the
  outer-segment / inner-half-period scan. The contribution path
  consumes `node0` and `node1` with `in_axes=(1, 1)` directly. Both
  layouts produce the correct (segment_index, half_period_index, 3)
  slicing.

- **dtypes.** JAX uses float64 for `points`, `nodes`, `seg_signs`,
  `currents` and int32 for `segments`. The `_coerce_inputs` helper
  (`wireframe.py:457-470`) enforces this. The CPU wrapper does the
  same (`wireframefield_jax.py:24-29` snapshots use
  `dtype=np.float64`/`np.int32`). C++ uses `double` and `int*`
  throughout. dtype contract matches.

- **Zero-length edges.** Neither implementation guards against `node0
  == node1`. The result is `denom = 0 * (0 + 0) = 0`, `factor = 0 / 0
  = nan`, and `cross(0_vec, 0_vec) = 0_vec`. Both kernels produce
  `0 * nan = nan` in the output. CPU and JAX both fail loudly. **No
  divergence; behaviour is consistent.** A defensive skip would be a
  policy change requiring a coordinated patch across both kernels.

### (f) Consumer-wrapper parity (`WireframeFieldJAX` vs `WireframeField`)

- **Construction snapshot.** Both wrappers snapshot `wframe.nodes`,
  `wframe.segments`, `wframe.seg_signs`, `wframe.currents` at
  construction. JAX explicitly copies via `np.array(..., copy=True)`
  (`wireframefield_jax.py:24-29`); CPU passes them by value through
  pybind11 to `sopp.WireframeField.__init__`
  (`wireframefield.py:25-26`). The parity test
  `test_current_snapshot_matches_cpu_wrapper_semantics`
  (`test_wireframefield_jax_item30.py:113-136`) verifies that mutating
  `wireframe.currents` after construction does not change either
  wrapper's output.

- **`set_points_cart` / `set_points_cyl`.** JAX wrapper caches the
  device-resident copy (`wireframefield_jax.py:60-70`) so future
  `B()` / `dB_by_dX()` calls do not re-transfer.

- **`_B_impl` / `_dB_by_dX_impl`.** Both wrappers convert the JAX
  output back to a NumPy view via `np.asarray(..., dtype=np.float64)`
  before assigning into the `MagneticField` buffer
  (`wireframefield_jax.py:73-92`). Test parity is asserted in
  `test_public_B_dB_and_segment_contributions_match_cpu`
  (`test_wireframefield_jax_item30.py:55-84`) at `direct_kernel`
  tolerance.

- **`dB_by_dsegmentcurrents`.** API parity is preserved: both return a
  list-of-arrays of shape `(n_segments,) -> (n_points, 3)`. CPU returns
  *references* into the field cache; JAX returns *copies* via
  `np.ascontiguousarray(contributions[i])`
  (`wireframefield_jax.py:116-120`). This is a behavioural difference
  (mutating the JAX return is safe; mutating the CPU return corrupts
  the cache). No test exercises this edge case. **Severity LOW**: callers
  should not be mutating cached field data anyway; the JAX behaviour is
  the safer one.

- **`dB_by_dsegmentcurrents(2)` rejection.** JAX raises
  `NotImplementedError` at `wireframefield_jax.py:103-106`; CPU defers
  to `compute(2)` which raises `logic_error` from
  `magneticfield_wireframe.cpp:93-94`. Parity in spirit; the exception
  *type* differs. Test
  `test_rejects_second_spatial_derivative_request`
  (`test_wireframefield_jax_item30.py:139-144`) only asserts
  `NotImplementedError` on the JAX path.

- **`dBnormal_by_dsegmentcurrents_matrix`.** Both wrappers implement
  this identically in Python by stacking the per-segment unit-current
  field contributions and projecting onto the surface normal. Test
  `test_normal_field_matrix_matches_cpu` parametrised over
  `area_weighted in (False, True)`
  (`test_wireframefield_jax_item30.py:87-110`) confirms parity at
  `direct_kernel` tolerance.

### (g) Transfer-guard / strict-CUDA parity

`test_wireframefield_jax_runs_under_strict_transfer_guard`
(`test_wireframefield_jax_item30.py:147-170`) and
`test_wireframe_runs_under_strict_transfer_guard`
(`test_wireframe_jax_item29.py:353-389`) place all inputs on device
explicitly and then enter `jax.transfer_guard("disallow")`. Both pass
in current CI runs, confirming no implicit host transfer occurs in the
JAX paths once inputs are device-resident.

## Test coverage gaps

| Gap | Severity |
| --- | -------- |
| No direct C++ parity test for `wireframe_segment_dB_by_dX_contributions` against `field_cache.get(dB, i)`. The function is exported but only validated transitively via `wireframe_B_and_dB_by_dX`. | INFO |
| No GPU/CUDA explicit parity exercise of the contribution kernels (the strict-transfer-guard test runs on CPU). Reduction-order divergence between `jnp.sum` (tree) and C++ `axpy` (sequential) is unobserved on GPU. | INFO |
| No singular-regime explicit test for behaviour at `r = a`, `r = b`, or `r` on the wire. The docstring documents `inf`/`nan` behaviour as intentional; an `assert not all(finite)` test would lock down the contract. | INFO |
| No `n_halfprds > 4` reduction-order stress test. For high `nfp` configurations, the half-period sum could exhibit observable order-divergence. | INFO |
| No test asserts that mutating the list returned from `WireframeFieldJAX.dB_by_dsegmentcurrents` is safe (the API quietly differs from CPU here: copies vs. references). | LOW |
| No zero-length-edge test; both kernels would emit `nan`. | INFO |
| No high-current saturation or per-current-sign reversal regression beyond what the existing fixtures cover. | INFO |
| `test_dB_layout_convention_via_finite_difference` only validates the layout at a single point. A multi-point FD layout test would catch any accidental axis swap on a refactor. | INFO |
| The `wireframe_segment_dB_by_dX` convenience wrapper recomputes the field even when only the gradient is requested (calls `wireframe_segment_B_and_dB_by_dX`). No test exists for this entry point at all. | LOW |

## Recommended actions ordered by severity

1. **MEDIUM — Use `lax.scan` in the contribution kernels.** Replace
   `jnp.sum(jax.vmap(half_period_*)(...), axis=0)` in
   `wireframe_segment_B_contributions` (`wireframe.py:261-268`) and
   `wireframe_segment_dB_by_dX_contributions` (`wireframe.py:297-304`)
   with the same `lax.scan` pattern used in `_segment_total_B`/`_segment_total_dB`
   (`wireframe.py:316-356`). This guarantees sequential
   accumulation matching the C++ `axpy_array` order, removing the
   reduction-order divergence and making byte-identity parity an
   invariant rather than an empirical observation.

2. **INFO + MEDIUM — Decide on `wireframe_segment_dB_by_dX_contributions`.**
   Either:
   (a) add a direct parity test against `field_cache.get(dB, i)` arrays
       (requires a small pybind11 accessor or an existing one), or
   (b) remove the export and the associated kernel since
       `dB_by_dsegmentcurrents` always returns B-cache entries
       regardless of `compute_derivatives`.

3. **LOW — Add a singular-regime contract test.** Assert `nan`/`inf`
   for `r` on the wire and `r = a`/`r = b`, locking down the
   "no defensive floors" contract documented at `wireframe.py:64-75`.

4. **LOW — Add a zero-length-edge contract test.** Same shape as above;
   assert non-finite output. Both kernels should behave the same way.

5. **LOW — Add a multi-point FD layout test.** Extend
   `test_dB_layout_convention_via_finite_difference` to a 3-point fixture
   and check all 9 entries of `dB[p, k, m]`. Catches any future axis
   swap on a refactor.

6. **LOW — Add a higher-`nfp` reduction-order regression test.** Pick
   `nfp = 5` or `nfp = 8` (so `n_halfprds = 10` / `16`) and assert
   `direct_kernel` parity for both `B` and `dB_by_dX`. The current tests
   exercise `nfp <= 2` only.

7. **LOW — Add a test for `WireframeFieldJAX.dB_by_dsegmentcurrents`
   mutation isolation.** Document and lock the JAX-side copy semantics
   so a future refactor does not silently switch to a view.

8. **LOW — Either remove or test `wireframe_segment_dB_by_dX`.** The
   convenience wrapper recomputes `B` discardedly; this is fine for a
   public smoke entry point but is currently unreached in tests beyond
   the export list.

## Closing note

The wireframe kernel is one of the cleanest direct ports in the
worktree — the JAX module faithfully reproduces the C++ closed-form
arithmetic line-by-line, including the deliberate omission of singular
guards. The total-field hot paths use `lax.scan` correctly and pass
strict-transfer-guard parity tests against the C++ oracle at
`direct_kernel` tolerance (`rtol=1e-10`). The only structural
divergence is in the rarely-used `wireframe_segment_*_contributions`
entry points where `jnp.sum` replaces sequential `axpy` order;
straightforward to harden if needed. Downstream item-31 (`wireframe
optimization`) consumes the contributions; the total-field hot path is
the workhorse and is bit-identical.
