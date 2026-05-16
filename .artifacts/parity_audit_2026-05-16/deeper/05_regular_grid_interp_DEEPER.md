# Parity Audit 05 — Regular-Grid Interpolant — DEEPER (second pass)

**Audit timestamp:** 2026-05-16 (second pass)
**Auditor scope:** PRIORITY 5 second-pass — hunting issues that forward-formula
parity audits systematically miss. Inputs: corrigendum of
`05_regular_grid_interp.md` and the first-pass HIGH/MEDIUM findings
(A: locator sign asymmetry; B: leave-result-unchanged semantics;
C: einsum-vs-FMA reduction order; nfp-fold subtlety).

---

## Executive summary — new findings

1. **HIGH — JIT-cache staleness on `spec.cell_table` mutation.** The
   `RegularGridInterpolant3DSpec` dataclass is `frozen=True`, but its
   NumPy fields (`cell_table`, `cell_to_row`, `xmesh`, `ymesh`, `zmesh`)
   are mutable (their `flags.writeable` is never cleared). The bare
   `evaluate_batch` path re-stages via `jnp.asarray(spec.cell_table, ...)`
   (`regular_grid_interp.py:587-589`) and **does** pick up mutations,
   but `interpolated_field_B / GradAbsB` use a one-time
   `_DeviceSpec` cache (`interpolated_field.py:346-381`) and **silently
   stay stale**. Two consumer paths over one spec disagree on the same
   underlying data. No test covers post-construction mutation.

2. **HIGH — int32 overflow in flat cell-index arithmetic.** JAX computes
   `flat_cell_idx = (xidx * ny + yidx) * nz + zidx` in int32
   (`regular_grid_interp.py:510-512, 521-523, 525`). For
   `nx*ny*nz > 2³¹-1 ≈ 2.1e9` the multiplication overflows silently and
   the gather hits an arbitrary row. C++ uses `int` for cell indices in
   `idx_cell` (`regular_grid_interpolant_3d.h:129-130`) so it has the
   same ceiling, but its UB on signed overflow is different from JAX's
   well-defined-mod silent corruption. The same overflow exists in
   `idx_dof` (`.h:124-127`) for the DOF table: for
   `nx=ny=nz=1000, degree=4`, `i*(ny*degree+1)*(nz*degree+1) =
   4000*4001*4001 ≈ 6.4e10`, computed in `int` → UB. Even the assignment
   to `uint32_t n` at `.h:228` cannot save it because the RHS is already
   UB before the cast.

3. **HIGH — Degree > 3 cross-oracle gap.** The wrapper at
   `interpolated_field_jax.py:185` pins `degree=4` and the test fixture
   at `test_interpolated_field_jax_item15.py:62` pins `_DEGREE = 4`. So
   degree=4 IS exercised transitively, but only through the
   symmetry-fold + Cartesian-rotation envelope on an analytic
   ToroidalField, where the Lagrange truncation residual is O(1e-9) — two
   decades above the `direct_kernel atol=1e-13` floor. The bare-kernel
   cross-oracle `test_cpp_cross_oracle` parametrises only
   `degree in [1, 2, 3]` (`test_regular_grid_interp_item13.py:276`).
   **Degree-4 reduction-order parity at the cell level is asserted only
   against a closed-form polynomial, never against the C++ kernel.**

4. **MEDIUM — `value_size != 1, 3` cross-oracle gap.** The C++ kernel
   pads `padded_value_size = round_up(value_size, simdcount)`. For
   `value_size=3, simdcount=4`, pad to 4 (33% memory overhead). For
   `simdcount=8 (AVX512)`, pad to 8 (167% overhead). JAX uses exact
   `value_size`. Numerical parity holds **only** because the padded
   columns of `AlignedPaddedVec local_vals(local_vals_size, 0.)`
   (`_impl.h:40`) are zero-initialised and never written. **No test pins
   this invariant on the C++ side.** Cross-oracle parametrises
   `value_size in [1, 3, 4]`; `5, 7, 8` (asymmetric SIMD padding) are
   absent.

5. **MEDIUM — Silent-and-wrong `jax.grad` through `evaluate_batch`.**
   The kernel is `jax.jit`-traceable AND `jax.grad`-traceable. The
   gradient is well-defined in the open interior of each cell (gradient
   of the piecewise polynomial), but it is **not** the same as the
   `GradAbsB` interpolant value. The two converge to the true gradient
   as `n → ∞` but differ at finite resolution by the interpolation
   residual. There is no blocker — a future caller doing
   `jax.grad(interpolated_field_B)(point)` gets a silently wrong answer.
   The CPU `field/tracing.py:99-106` wrapper already synthesises
   `dB_by_dX = grad_abs_B ⊗ B / |B|`, a rank-1 approximation that is
   correct for guiding-centre vacuum dynamics but wrong for full
   `dB/dX` queries.

6. **MEDIUM — NaN-in / NaN-out is undefined on C++ side.** Tracing
   `x = NaN` through JAX yields `xidx_raw = floor(NaN).astype(int32) =
   INT32_MIN` on x86 XLA, then `in_bounds = False`, sentinel-row gather,
   `xlocal = NaN`, einsum returns NaN. **C++ `int(NaN)` is
   implementation-defined** (typically `INT_MIN` on x86 `cvttsd2si`), so
   `idx_cell(INT_MIN, INT_MIN, INT_MIN)` overflows and the
   `unordered_map::find` call has undefined input. With
   `out_of_bounds_ok=True` C++ may silently return unchanged, segfault,
   or corrupt. JAX surfaces NaN consistently. **This bites tracing
   exactly when the RHS produces NaN (turning-point banana orbits,
   first-pass row 4).**

7. **LOW — Classifier short-circuit is preserved on GPU and under
   `vmap`, but is closure-based, not JIT-decorated.** The
   `signed_distance_to_cartesian_classifier` returns a bare Python
   closure (`surface_classifier.py:72-103`). Inside `jax.vmap` the
   `was_single` branch evaluates False (input ndim grows by 1) and the
   short-circuit applies elementwise — correct. **First call after
   `vmap` recompiles the inner `_evaluate_batch_jit`.** No GPU-specific
   divergence.

8. **LOW — `_DeviceSpec` lifetime.** Each `InterpolatedFieldJAX`
   holds `_device_B` + `_device_GradAbsB`. For a production grid
   `(r=64, phi=128, z=64), degree=4, value_size=3`,
   `cell_table` is `524288 * 125 * 3 * 8 ≈ 1.5 GB` per spec. JIT compile
   caches may pin references after the wrapper is released. No leak
   guard.

---

## A — Degree-4 reduction-order coverage

### A.1 — Where degree ≥ 4 actually runs

| Test | Coverage |
|------|----------|
| `test_regular_grid_interp_item13.py::test_polynomial_exactness[degree=4]` (L70-105) | JAX vs **closed-form**, no C++ |
| `test_regular_grid_interp_item13.py::test_cpp_cross_oracle` (L275-326) | JAX vs C++, only `degree in [1, 2, 3]` |
| `test_interpolated_field_jax_item15.py::test_in_domain_B_parity` | wrapper-level, `_DEGREE=4` (L62) |
| `tests/field/test_interpolant.py::test_regular_grid_interpolant_exact` | C++ vs closed-form, `degree in [1,2,3,4]` |

### A.2 — Why the wrapper test under-covers degree-4 kernel parity

Wrapper test compares JAX (kernel+fold+rotate) to C++ (kernel+fold+rotate)
on `ToroidalField`. The Lagrange degree-4 interpolant residual against
`1/r` on `_R_RANGE = (1.0, 1.5, 8)` is empirically O(1e-9). The cell-level
einsum-vs-FMA ULP divergence is O(1e-13). The truncation residual hides
the kernel divergence by four orders of magnitude. **The wrapper test
passes at `direct_kernel = (rtol=1e-10, atol=1e-13)` regardless of
kernel-level parity at degree 4.**

### A.3 — When reduction order becomes load-bearing

For `degree=4, value_size=3`, the einsum sums `5³ = 125` products. For
cell-data values O(1e3–1e6) (high-current coils, integral decompositions
of near-axis quantities), the accumulated ULP can grow to O(1e-7)–O(1e-9),
blowing the `direct_kernel atol=1e-13` floor. **The current ToroidalField
fixture happens to keep cell data at O(1), masking the failure mode.**

**Recommend:** add `degree in [4, 5]` to `test_cpp_cross_oracle`. Add a
fixture with cell-data values O(1e5) to expose the einsum-vs-FMA path
at high degree.

---

## B — `value_size != 1, 3` coverage and SIMD padding

### B.1 — C++ padding mechanism

`regular_grid_interpolant_3d.h:295`:
```cpp
padded_value_size = (value_size % simdcount) ? (value_size + simdcount) - (value_size % simdcount) : value_size;
```

- `value_size=3, simdcount=4 (AVX)` → padded=4
- `value_size=3, simdcount=8 (AVX512)` → padded=8
- `value_size=5, simdcount=4` → padded=8
- `value_size=8, simdcount=8` → padded=8

The inner loop iterates `for(int l=0; l<padded_value_size; l += simdcount)`,
producing `padded` outputs from `padded * (degree+1)³` cell values.
Only the first `value_size` outputs are written to `res[l+ll]`
(`_impl.h:170`); the rest are computed-and-discarded. **Correctness depends
on the padded cell-data columns being zero**, which holds because
`AlignedPaddedVec(local_vals_size, 0.)` initialises to zero
(`_impl.h:40`) and only the first `value_size` columns are written by
the assignment at `_impl.h:46-48`.

### B.2 — Risk: padded-tail invariant

If a future C++ change ever writes to the padded tail (e.g. an
asymmetric SIMD masking trick that reuses upper lanes for partial sums),
the C++ result diverges from JAX. **There is no test that pins the
padded-tail invariant.** A black-box test that catches it:
build a degree-1 single-cell interpolant with `value_size=1` and inject
corner DOFs `[1, 0, 1, 0, 1, 0, 1, 0]`; only correct if padded tail is
ignored.

### B.3 — Cross-oracle gap

| value_size | Cross-oracle | SIMD case |
|------------|--------------|-----------|
| 1 | yes | padded=4 or 8 (extreme padding) |
| 3 | yes | padded=4 (slight padding) |
| 4 | yes | padded=4 (no padding on AVX) |
| 5 | **NO** | padded=8 (asymmetric AVX) |
| 6 | partial (C++ closed-form only) | padded=8 |
| 7, 8 | **NO** | padded=8 (8 hits SIMD boundary) |

Recommend extending `test_cpp_cross_oracle` parametrisation.

---

## C — Symmetry-fold edge cases (nfp boundary)

### C.1 — C-style `int(phi/period)` re-analyzed

`magneticfield_interpolated.h:105-106`:
```cpp
int phi_mult = int(phi/period);
phi = phi - phi_mult * period;
```

For `phi >= 0`, equivalent to `floor(phi/period)`. For `phi < 0`,
truncates toward zero, leaving `phi - 0*period = phi < 0` (still negative).

JAX (`interpolated_field.py:132-133`):
```python
phi_wrapped = jnp.where(phi < 0.0, phi + 2.0 * jnp.pi, phi)
return jnp.mod(phi_wrapped, period)
```

Explicitly maps negatives to `[0, 2π)` before `jnp.mod`, which returns
`[0, period)`.

**Divergence case:** `arctan2` returns `(-π, π]`. With `nfp=1,
stellsym=False`, no symmetry fold is applied at the wrapper level:
- CPU `exploit_symmetries_points` runs even when `nfp=1 && !stellsym`,
  doing `phi_mult = int(phi/(2π)) = 0` for phi in `(-π, π]`. So `phi`
  stays in `(-π, π]`.
- The C++ rectangular kernel then sees `phi < 0`, soft-clamp
  `phi <= ymin → phi += 1e-13` doesn't help (assuming `ymin=0`).
  `int(ny * negative / period) = 0` (C-truncation toward zero).
  With `out_of_bounds_ok=True`, falls through to `evaluate_local(cell_idx_with_xidx_0...)`.
  Effectively extrapolates from cell 0 into negative phi.
- JAX folds explicitly via `_fold_phi_nfp`, so negative phi never
  reaches the JAX rectangular kernel.

**Outcome:** for nfp=1 without stellsym, C++ silently extrapolates from
cell 0 for negative `arctan2` outputs; JAX folds first. With
`out_of_bounds_ok=False` C++ throws (because `phi - 1e-13` doesn't help
when the input is `-π/4`). The wrapper test fixture at
`_PHI_RANGE_FULL = (0.0, 2π, 16)` happens to cover `[0, 2π]` so the
negative-phi case after `arctan2` IS exercised — but the wrapper's
`InterpolatedField` constructor passes phi range to the C++ kernel
which then handles it.

Inspecting the C++ path more carefully: with phi_range = (0, 2π) and
nfp=1, the C++ wrapper still calls `exploit_symmetries_points` which
folds `phi = phi - int(phi/2π) * 2π`. For `phi = -π/4`,
`int(-0.125) = 0`, `phi = -π/4 - 0 = -π/4` (still negative). The C++
kernel then has `phi_min = 0`. `phi - phi_min = -π/4 < 0`. Soft clamp
won't fire. **C++ silently extrapolates with `out_of_bounds_ok=True`.**
JAX correctly folds to `7π/4`. Both then produce a value, but it is
the value at a different cell. **This compounds first-pass finding A.**

### C.2 — `phi` exactly at the period boundary

For `phi = 2π/nfp - epsilon` (just below upper boundary): both kernels
agree, cell `ny-1`. For `phi = 2π/nfp` exactly: both clamp to
`2π/nfp - 1e-13`, cell `ny-1`. For `phi = 2π/nfp + epsilon`: both fold
via the wrapper to small positive, cell 0.

The genuinely risky case is `phi_pre = 2π - tiny` AFTER stellsym
reflection of `phi_original = tiny`. Both kernels handle this case
correctly: C++ `int(2π/period)` for `nfp ≥ 2` and `period = 2π/nfp`
gives `nfp` exactly (or `nfp - 1` if 2π/period rounds down due to FP
noise), so `phi = (2π - tiny) - nfp * (2π/nfp) = -tiny` (slightly
negative) — which then triggers finding A. JAX `jnp.mod((2π - tiny),
period)` returns `period - tiny` (close to period from below) which is
safely in-range.

**Witness:** `tests/field/test_interpolated_field_jax_item15.py:286-318`
covers the combined fold with margins `0.02` away from boundaries; the
genuinely tight boundary case is NOT exercised.

---

## D — Cell-locator at upper boundary

Sweep through the upper-boundary edge cases:

| Input | C++ | JAX | Parity |
|-------|-----|-----|--------|
| `x = xmax` | clamp → `xmax-1e-13` → cell `nx-1` | same | OK |
| `x = xmax + 1e-13` | clamp → `xmax` → `int(nx)` → OOB | clamp → `xmax` → `floor(nx) = nx` → OOB | OK |
| `x = xmax + 1e-14` | clamp → `xmax - 9e-14` → cell `nx-1` | same | OK |
| `x = xmax - 1 ULP` | no clamp, `int(nx*(1-2ulp))` = `nx-1` | same | OK |
| `x = xmax + 1e-12` (above eps) | no clamp, `int(nx*(1+ε))` = `nx` → OOB | `floor(nx*(1+ε)) = nx` → OOB | OK |

**Upper boundary is symmetric.** The first-pass finding A applies only
at the lower boundary. Lower-boundary sweep:

| Input | C++ | JAX | Parity |
|-------|-----|-----|--------|
| `x = xmin` | clamp → `xmin + 1e-13` → cell 0 | same | OK |
| `x = xmin - 1e-14` | clamp → `xmin + 9e-14` → cell 0 | same | OK |
| `x = xmin - 1e-12` | no clamp, `int(nx*(-2e-12))` = `int(-1.6e-11) = 0` (C-trunc) | no clamp, `floor(-1.6e-11) = -1` (FLOOR) | **DIVERGENT — Finding A reproduced** |

---

## E — NaN propagation through `evaluate_batch`

Tracing `x = NaN`:

| Step | JAX | C++ |
|------|-----|-----|
| Soft clamp `x >= xmax` | `NaN >= xmax = False` → no clamp | same |
| `xidx_raw = floor(NaN*...).astype(int32)` | XLA: `INT32_MIN` on x86 | `int(NaN*...)` impl-defined; x86 `cvttsd2si` returns `INT_MIN` |
| `in_bounds_x = (INT_MIN >= 0)` | `False` | (no equivalent check) |
| `xidx = clip(INT_MIN, 0, nx-1)` | `0` | (no clip) |
| Cell lookup | sentinel row | `idx_cell(INT_MIN, ...)` → arbitrary, UB |
| `xlocal = (NaN - xmesh[0])/hx` | `NaN` | `NaN` |
| `einsum` / SIMD inner | `NaN` (NaN×0=NaN) | depends on `find()` outcome |
| Result | `NaN` consistently | `out_of_bounds_ok=True`: silent miss or UB; `False`: throw |

**Critical:** JAX `out_of_bounds_ok=True` returns NaN (not zero) when
the *input* is NaN, even though the first-pass finding B documented
zero on clean OOB. So the OOB-OK semantic is **"zero on clean OOB, NaN
on NaN input"**. C++ is UB.

**Bite point:** the JAX RHS at `jax_core/tracing.py` calls
`magnetic_field_fn(y)` for a state `y` that may have a NaN
position (turning-point banana orbits per first-pass row 4). The JAX
path then produces NaN, which **silently propagates through the ODE
integrator** unless explicitly checked. The CPU path may throw, segfault,
or return garbage. Tests do not cover NaN-input behaviour.

---

## F — Autodiff through `evaluate_batch`

### F.1 — Differentiability profile

`_evaluate_batch_jit` is differentiable w.r.t. `xyz` in the open
interior of each cell:
- `jnp.floor(...).astype(int32)`: gradient is zero (constant index).
- `xlocal = (x_clamped - xmesh[xidx])/hx`: gradient is `1/hx`.
- `_basis_values`: smooth in `xlocal` (degree-d polynomial).
- `cell_table[row_idx]`: gradient is zero w.r.t. xyz.
- `jnp.einsum`: smooth in `pkx, pky, pkz`.

At a cell boundary, the gradient has a step discontinuity (O(1/hx)
times the cell-data jump). `jax.grad` returns the left-side or
right-side gradient depending on `xidx`'s clamping side.

### F.2 — The wrong-physics trap

`InterpolatedFieldJAX` exposes two paths:
1. `jax_B_at(point)` → `interpolated_field_B(spec, point)`.
2. `jax_B_GradAbsB_at(point)` → returns `(B, ∇|B|)` via two separate
   interpolant tables.

If a caller does `jax.grad(jax_B_at)(point)`, JAX differentiates the
**interpolant of B**. The result is `d(Lagrange_4(B))/dx`, a piecewise
polynomial of degree 3. The correct quantity for physical gradients is
`Lagrange_4(dB/dx)`, supplied by the `GradAbsB` interpolant table that
was sampled from the analytic gradient at construction time. These two
are **provably different** at finite resolution:

  d/dx of degree-4 Lagrange interpolant of f ≠ degree-4 Lagrange interpolant of df/dx

There is no blocker. No test catches accidental use of the wrong path.

**Compounding:** `field/tracing.py:99-106` already synthesises
`dB_by_dX = grad_abs_B[:, None] * B[None, :] / abs_B`. This rank-1
matrix is the projection onto B-hat — mathematically sufficient for
`gc_vac` dynamics but wrong for full `dB/dX` queries. **Three sources
of "gradient" exist with subtly different semantics:**
- `jax.grad(jax_B_at)` → gradient of the interpolant (mathematically a polynomial).
- `jax_B_GradAbsB_at()` → interpolant of the gradient (mathematically correct).
- `_field_fn` in `tracing.py:99-106` → rank-1 projection from `GradAbsB` and `B`.

---

## G — JIT-cache staleness on spec mutation

### G.1 — Mutation surface

`@dataclass(frozen=True)` prevents reassignment but NOT in-place mutation
of NumPy fields:

```python
spec = build_regular_grid_interpolant_3d(...)
spec.cell_table[0, 0, 0, 0] = 999.0  # ALLOWED, no error
```

### G.2 — Path 1: `evaluate_batch`

`regular_grid_interp.py:587-590` re-stages via `jnp.asarray(spec.cell_table)`.
Picks up mutations.

### G.3 — Path 2: `interpolated_field_B/GradAbsB`

`interpolated_field.py:89-90` builds `_DeviceSpec` **once** at construction.
Subsequent calls reuse the cached device array. **Mutations to
`spec.B_spec.cell_table` are NOT propagated.**

### G.4 — Risk surface

Two consumer paths of one spec object disagree on the same underlying
data after any mutation. A future contributor adding a "refresh the
spec" method would see `evaluate_batch` reflect the change, conclude
the design works, and ship — while `interpolated_field_B` silently uses
stale data.

### G.5 — Fix

Add `__post_init__` to `RegularGridInterpolant3DSpec`:
```python
def __post_init__(self):
    for arr in (self.cell_table, self.cell_to_row,
                self.xmesh, self.ymesh, self.zmesh):
        arr.flags.writeable = False
```

Add a test: `with pytest.raises(ValueError): spec.cell_table[0, 0, 0, 0] = 1.0`.

---

## H — C++ UB hunt

| Site | Concern |
|------|---------|
| `regular_grid_interpolant_3d.h:124-127` `idx_dof(i,j,k) = i*(ny*degree+1)*(nz*degree+1) + ...` | Returns `int`. For `nx=ny=nz=1000, degree=4`, evaluates to `6.4e10`, overflows `INT_MAX = 2.1e9`. **UB.** |
| `regular_grid_interpolant_3d.h:133-135` `idx_mesh` | Same overflow for `nx=ny=nz=1500`. **UB.** |
| `regular_grid_interpolant_3d.h:228` `uint32_t n = (nx*degree+1)*(ny*degree+1)*(nz*degree+1)` | RHS evaluated in `int` then assigned to `uint32_t`. The `int` computation is UB before the cast. |
| `regular_grid_interpolant_3d.h:115` `pkxs, pkys, pkzs` member buffers mutated in `evaluate_local` | Serial loop now; thread-unsafe if anyone parallelises `evaluate_batch`. Latent race. |
| `_impl.h:64` `fxyz.data() + value_size*i` | `value_size*i` in `int`. For `value_size=3, i > 7e8`, overflows. Latent practical ceiling at ~7e8 points. |
| `_impl.h:60-61` row-major check on `fxyz` but not on `xyz` | Functionally fine but performance asymmetric for column-major xyz input. |

No new aliasing bugs found beyond the first-pass coverage.

---

## I — Memory parity / padded tail

| Setting | C++ per-cell bytes | JAX per-cell bytes | Δ |
|---------|--------------------|--------------------|----|
| degree=4, value_size=3, simdcount=4 (AVX) | 4000 | 3000 | +33% C++ |
| degree=4, value_size=3, simdcount=8 (AVX512) | 8000 | 3000 | +167% C++ |
| degree=2, value_size=1, simdcount=8 | 216 | 27 | +700% C++ |

**No boundary risk** because the cell table is private to the C++
binding; the Python boundary always sees exact `value_size`. Internal
memory amplification only.

---

## J — `estimate_error` RNG

C++ uses default-constructed `std::default_random_engine` (LCG, default
seed 1). JAX uses `np.random.default_rng(seed=0)` (PCG64). Brackets are
different by construction. The JAX test
(`test_estimate_error_returns_bracket_for_polynomial`) asserts
`|low|, |high| < first_derivative_atol` only — passes silently for any
polynomial-exact case regardless of RNG.

**No hidden parity claim** because the JAX test is JAX-only. **No
positive cross-oracle evidence** either. Recommend pinning each
kernel's bracket against a documented expected value (different per
kernel; do NOT compare).

---

## K — Skipped-cell map performance

| Skip density | C++ map | JAX gather |
|--------------|---------|------------|
| 0% (no skip) | O(1) per lookup, poor cache (heap buckets) | O(1) coalesced gather |
| 50% | O(1), 50% miss-cheap | same gather, 50% sentinel |
| 99% | small kept set, hash buckets sparse | sentinel-row dominates, cache-friendly |

Behavioural parity is identical. Performance favours JAX on GPU
(coalesced). No new finding.

---

## L — Classifier short-circuit on GPU / under vmap

`surface_classifier.py:90-97`:
```python
in_bounds = (r >= xmin) & (r <= xmax) & (phi >= ymin) & ...
result = jnp.sign(jnp.where(in_bounds, dist_flat, -1.0))
```

Pure `jnp` arithmetic, no CPU-specific code. Compiles to XLA `select`
on GPU. Under `vmap`, `in_bounds` becomes a leading-axis batched
boolean and `jnp.where` operates elementwise. Preserved on GPU and
under `vmap`. The first call after `vmap` recompiles the inner
`_evaluate_batch_jit` due to shape signature change; subsequent calls
hit cache.

The classifier short-circuit applies only to the **classifier closure**,
not to `evaluate_batch` directly. Callers that bypass the classifier
get raw kernel behaviour (NaN/zero on OOB).

---

## Untested edge-case inventory

| Edge case | Reproduce | Severity |
|-----------|-----------|----------|
| `degree=4, value_size=3` bare-kernel cross-oracle | parametrize `test_cpp_cross_oracle` | HIGH |
| `degree=5` bare-kernel cross-oracle | same | HIGH |
| `value_size in [5, 7, 8]` cross-oracle | same | MEDIUM |
| `nx*ny*nz > INT32_MAX` overflow guard | unit test, expect raise | HIGH |
| NaN input | `evaluate_batch(spec, [[NaN, ymid, zmid]])` | MEDIUM |
| Post-construction `spec.cell_table` mutation | mutate, re-evaluate via both paths | HIGH |
| `jax.grad(jax_B_at)` differs from `jax_B_GradAbsB_at` | assert ≠ | MEDIUM |
| Chebyshev cross-oracle | parametrize `test_cpp_cross_oracle` over `rule_factory` | LOW |
| `estimate_error` per-kernel pinned bracket | new C++ + JAX tests | LOW |
| C++ padded-tail invariant | black-box `value_size=1` degree-1 stress test | MEDIUM |
| `x = xmin - 1e-12` lower-boundary parity | finding-A reproducer | HIGH (from first-pass) |
| `phi` exactly `2π/nfp` after stellsym reflection | wrapper boundary test | MEDIUM |
| Empty domain (`cells_to_keep == 0`) | exists in code; no test | LOW |

---

## Recommended actions ordered by severity

### HIGH

1. **Spec immutability.** `RegularGridInterpolant3DSpec.__post_init__`
   sets `flags.writeable = False` on every NumPy field. Test:
   `with pytest.raises(ValueError): spec.cell_table[0] = 1.0`.
2. **Int32 overflow guard.** In `_build_cell_table`:
   ```python
   if max(nx*ny*nz, (nx*degree+1)*(ny*degree+1)*(nz*degree+1)) > 2**31 - 1:
       raise OverflowError(...)
   ```
3. **Degree-4 cross-oracle.** Extend `test_cpp_cross_oracle`
   parametrisation to `degree in [4, 5]`. If it fails, swap einsum for
   `jax.lax.fori_loop` matching C++ reduction order.

### MEDIUM

4. **NaN parity contract.** Add a JAX-side test that asserts NaN-in →
   NaN-out for `out_of_bounds_ok in [True, False]`. Document C++ side
   as UB; mark as `xfail`.
5. **Autodiff blocker / docstring.** Doc-warn at
   `interpolated_field_jax.py:222` (`jax_B_at`) that `jax.grad` returns
   the gradient of the interpolant, not the physical gradient. Point to
   `jax_B_GradAbsB_at`.
6. **Padded-tail invariant test.** Black-box test on C++ binding with
   `value_size=1, degree=1, simdcount=8`.
7. **`value_size in [5, 8]` cross-oracle.** Extend parametrisation.

### LOW

8. **Sparse-skip benchmark.** 1%, 50%, 99% skip density; pin ratio at
   `≤ 2x` regression guard.
9. **`estimate_error` per-kernel pin.** Separate JAX and C++ tests
   each pinning a fixed expected bracket; do not compare.
10. **`_DeviceSpec` lifetime.** Document; add weakref test if JAX
    semantics allow.

---

## Cross-references for downstream priorities

- **Priority 4 (tracing).** Section F (silent-wrong-physics `jax.grad`)
  and Section E (NaN UB on C++) are critical for any "differentiable
  particle tracing" or NaN-handling workflow. Document NOW.
- **Boozer interpolant** (`interpolated_boozer_field.py`). Inherits
  Sections G (mutation staleness), H (C++ UB), and A (degree>3 gap).
  16+ specs per field instance → memory analysis (Section I)
  multiplies by 16+.
- **Classifier.** Section L confirms CPU short-circuit
  (`surface.py:90-97`) is preserved on GPU and under `vmap`. No
  GPU-specific divergence.
