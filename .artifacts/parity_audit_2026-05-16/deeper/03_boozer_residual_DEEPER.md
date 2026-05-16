# Second-Pass Boozer Residual JAX Audit — Deeper Edge Cases

**Target**: `src/simsopt/geo/boozer_residual_jax.py` (802 lines) vs `src/simsoptpp/boozerresidual_impl.h`, `boozerresidual_py.cpp`/`h`.

**Cross-refs**: `src/simsopt/geo/surfaceobjectives.py::boozer_surface_residual`, `src/simsopt/geo/boozersurface.py::boozer_penalty_constraints_vectorized`, `src/simsopt/geo/boozersurface_jax.py`, `src/simsopt/geo/label_constraints_jax.py::compute_G_from_currents`, `tests/geo/test_boozer_residual_jax.py`, `tests/geo/test_boozer_derivatives_jax.py`.

**Scope**: This is a second-pass audit. Forward-formula parity was already validated by the first pass. We hunt only for findings the forward-parity audit would not have caught: edge cases, dtype footguns, decision-vector packing, JIT closure / cached state risk, autodiff edge cases (jacfwd vs jacrev memory), VJP correctness for fixed coils + weight_inv_modB, API surface mismatches, C++ UB patterns, physics convention drift, and test fixture realism.

---

## Executive Summary

Eight distinct findings, ordered by severity:

| ID | Severity | Title | Locus |
|----|----------|-------|-------|
| D1 | HIGH | `_split_decision_vector` silently truncates surface DOFs on shape mismatch (no bounds check) | `boozer_residual_jax.py:73-82` |
| D2 | HIGH | `_unpack_decision_vector` G-from-currents fallback duplicates `compute_G_from_currents` instead of importing it (two SSOTs for the same physics constant) | `boozer_residual_jax.py:554-556` vs `label_constraints_jax.py:49-61` |
| D3 | MEDIUM | `boozer_residual_jacobian_composed` always uses `jax.jacfwd` — no `n_res` vs `n_dofs` heuristic; memory blows up at moderate `(nphi*ntheta, mpol, ntor)` (LFD/RAM scaling is `n_dofs · n_res · 8B`) | `boozer_residual_jax.py:717-740` |
| D4 | MEDIUM | `boozer_residual_jacobian_composed` recomputes residual twice (once for `r`, once inside `jacfwd`) — first-pass F2 flagged but JAX-side closure already double-traces because kwargs splat re-evaluates | `boozer_residual_jax.py:738-740` |
| D5 | MEDIUM | `boozer_residual_coil_vjp` traces through `weight_inv_modB=True` correctly via autodiff, but defaults to `False` while `boozer_residual_vector` defaults to `True` — silent contract drift between siblings | `boozer_residual_jax.py:752, 293` |
| D6 | MEDIUM | `_inverse_modB` returns NaN at |B|=0; the scalar pipeline does NOT short-circuit and propagates NaN into both objective and gradient | `boozer_residual_jax.py:85-87, 95-96` |
| D7 | LOW | dtype-mixing footgun: `_as_runtime_float64` ignores the float32 reference and silently promotes to float64; the user's float32 input gets demoted without warning, then numpy returns the float32 dtype-promotion error stack only if `_as_jax_array` raises | `_math_utils.py:61-69` |
| D8 | LOW | C++ kernel has no `pragma omp` (so no race possible) and braces are correct — no UB pattern recurrence. However, AlignedPaddedVec `ndofs+2` arrays are SIMD-loaded for the Hessian outer product, relying on `ndofs+2` being padded mod `simd_size`; if a future allocator change drops padding, this becomes a read past the end | `boozerresidual_impl.h:29-47, 205-208` |
| T1-T6 | LOW | Test fixture gaps: tests use `nphi=ntheta=4-10` and `mpol=ntor=1` only; no non-stellsym multi-helicity or sign-flip fixtures; no iota=0 or zero-current edge cases; no float32 dtype probe; no jacfwd-vs-jacrev cross-check | `test_boozer_residual_jax.py`, `test_boozer_derivatives_jax.py` |

No critical findings (no silent wrong-answer paths under default kwargs). D1, D2 are corner cases that escape because of weak input validation but do not currently bite production callers (verified inside `boozersurface_jax.py`, all consumers construct `x` from `_pack_optimizer_state`, never directly).

---

## D1 — HIGH: `_split_decision_vector` has no shape validation

**File**: `src/simsopt/geo/boozer_residual_jax.py:73-82`

```python
def _split_decision_vector(x, *, optimize_G):
    x_jax = _as_jax_float64(x)
    tail_size = 2 if optimize_G else 1
    surface_size = int(x_jax.shape[0]) - tail_size
    sdofs = jnp.take(x_jax, _as_jax_int32(np.arange(surface_size)), axis=0)
    iota = jnp.take(x_jax, _as_jax_int32(surface_size), axis=0)
    if optimize_G:
        G = jnp.take(x_jax, _as_jax_int32(surface_size + 1), axis=0)
        return sdofs, iota, G
    return sdofs, iota, None
```

### Pitfalls

1. **Negative `surface_size` is silently allowed.** If `optimize_G=True` and someone passes a length-1 vector, `surface_size = -1`. `np.arange(-1)` returns an empty array. `jnp.take(x, [], axis=0)` returns an empty array. Then `jnp.take(x, -1)` returns the last element. The caller gets `(empty_sdofs, x[-1], x[0])` — silently wrong.

2. **No assertion that `x.shape[0] >= tail_size`**. The CPU counterpart has implicit guards via `dofs[:-2]` slicing — a length-1 dofs vector under `optimize_G=True` would yield `sdofs.size==0` then `dofs[-1]` indexes last element and `dofs[-2]` IndexErrors. JAX silently produces wrong shapes with `jnp.take`.

3. **Inconsistent indexing into JAX shape under tracing.** Calling `int(x_jax.shape[0])` is fine for non-tracing but **invokes the eager shape at JIT trace time only**. Under `vmap` or other transforms that abstract the shape, this raises `ConcretizationTypeError`. Production callers route through cached entrypoints with fixed `optimize_G`, so this hasn't bitten anyone — but if a future caller jit-traces with `optimize_G` as a JAX argument, this silently breaks.

### Recommendation

Add explicit validation at entry:

```python
def _split_decision_vector(x, *, optimize_G):
    x_jax = _as_jax_float64(x)
    tail_size = 2 if optimize_G else 1
    surface_size = int(x_jax.shape[0]) - tail_size
    if surface_size < 0:
        raise ValueError(
            f"Decision vector length {x_jax.shape[0]} too short for optimize_G={optimize_G}; "
            f"requires at least {tail_size} tail entries."
        )
    ...
```

### Untested edge cases

- `optimize_G=True` with `x.shape == (1,)` or `(2,)` (currently produces wrong shapes silently).
- `optimize_G=False` with `x.shape == (0,)`.
- `x.shape == ()` (rank-0 scalar input).

---

## D2 — HIGH: G-from-currents formula is duplicated, not referenced

**File**: `src/simsopt/geo/boozer_residual_jax.py:554-556`

```python
all_currents = jnp.concatenate([c for _, _, c in coil_arrays])
mu0 = _as_runtime_float64(4.0e-7 * np.pi, reference=all_currents)
return sdofs, iota, mu0 * jnp.sum(jnp.abs(all_currents))
```

vs. **`src/simsopt/geo/label_constraints_jax.py:49-61`** (the SSOT per CLAUDE.md):

```python
def compute_G_from_currents(currents):
    """G = 2π Σ|I_k| · μ₀/(2π) = μ₀ Σ|I_k|"""
    mu0 = 4.0 * jnp.pi * 1e-7
    return mu0 * jnp.sum(jnp.abs(currents))
```

### Why this matters

The SSOT lives in `compute_G_from_currents` (used by 12 sites in `boozersurface_jax.py` and `surfaceobjectives_jax.py`). The duplicate inline expression in `_unpack_decision_vector` is one bug-fix away from divergence. The C++ oracle in `surfaceobjectives.py:578-584` uses yet a third spelling:

```python
G = 2.0 * np.pi * np.sum(np.abs([currents])) * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
```

All three currently algebraically reduce to `μ₀ · Σ|I_k|`, but:

- `compute_G_from_currents` uses `4.0 * jnp.pi * 1e-7` (JAX π in float64, multiplied at runtime — no NumPy-vs-JAX rounding identity across hardware).
- `boozer_residual_jax.py:555` uses `4.0e-7 * np.pi` (NumPy π in float64, evaluated host-side once via Python float arithmetic, then promoted via `_as_runtime_float64`).

Under strict bit-identity, `np.pi` and `jnp.pi` are the same float64 value, but a future drift (e.g., changing one to `math.pi`, or moving to a different precision policy) could silently desync only this one site without any test catching it (because `compute_G_from_currents` lives behind a different cache key).

### Recommendation

Replace lines 554-556 with:

```python
from .label_constraints_jax import compute_G_from_currents

all_currents = jnp.concatenate([c for _, _, c in coil_arrays])
return sdofs, iota, compute_G_from_currents(all_currents)
```

(There's a circular import risk because `label_constraints_jax` imports from elsewhere in the geo subpackage; if so, do a late import inside the function to match the existing lazy-import pattern at lines 449-483.)

### Untested edge case

- **Zero coil currents**: if all currents are zero and `optimize_G=False`, then `G = 0`. The Boozer residual `G·B - |B|² · tang` then degenerates to `-|B|² · tang`, which is the Boozer-violation magnitude in a current-free shell. No test exercises this; in practice it would surface as a divergent LS solve since the residual cannot reach zero except on the (degenerate) curve where `tang` ⊥ `B`.

---

## D3 — MEDIUM: Jacobian always uses `jax.jacfwd`; no memory guard

**File**: `src/simsopt/geo/boozer_residual_jax.py:717-740`

```python
def boozer_residual_jacobian_composed(x, **kwargs):
    r = _boozer_residual_vector_composed(x, **kwargs)
    J = jax.jacfwd(_boozer_residual_vector_composed)(x, **kwargs)
    return r, J
```

### Why `jacfwd` is wrong here in general

The residual vector has size `n_res = 3 * nphi * ntheta`. The decision vector has size `n_dofs = nsurfdofs + 1 + optimize_G`.

- `jacfwd` costs `O(n_dofs)` forward passes through the pipeline. For each forward pass, the pipeline materializes a full `(nphi, ntheta, 3)` `B` array plus geometry derivatives.
- `jacrev` costs `O(n_res)` reverse passes. For typical Boozer LS shapes (`nphi=32, ntheta=32`, so `n_res = 3072`), `n_res >> n_dofs` and `jacfwd` is correct.
- BUT: in **strict CPU/JAX byte-identity** lanes where the test fixtures shrink to `nphi=ntheta=8` (i.e., `n_res=192`) with `mpol=4, ntor=4` (i.e., `n_dofs ~= 243`), the choice inverts.

The C++ kernel `sopp.boozer_dresidual_dc` builds the Jacobian in a hand-rolled forward-mode fashion, which is also `O(n_dofs)`-memory but uses cache-friendly SIMD over `m+=simd_size` chunks. The JAX `jacfwd` allocates the full `(n_res, n_dofs)` matrix plus per-coordinate intermediate `(n_res, n_dofs, ...)` tensors during tracing.

### Memory budget

For `n_res=3072, n_dofs=243`, `J` is ~6 MB. Tractable. For `n_res=12288 (nphi=64), n_dofs=2401 (mpol=ntor=24)`, `J` is ~236 MB plus intermediate cotangent buffers — exceeds GPU register/SM cache and goes to global memory; large slowdown and potential OOM.

### No fallback path

There is no `n_res < n_dofs ⇒ jacrev` heuristic. A user shrinking `nphi*ntheta` to investigate behavior at low-`n_res` regimes pays the wrong-direction cost.

### Recommendation

Add a heuristic guard:

```python
def boozer_residual_jacobian_composed(x, **kwargs):
    n_dofs = int(x.shape[0])
    # Run residual once, infer n_res, choose direction.
    r = _boozer_residual_vector_composed(x, **kwargs)
    n_res = int(r.shape[0])
    if n_res >= n_dofs:
        J = jax.jacfwd(_boozer_residual_vector_composed)(x, **kwargs)
    else:
        J = jax.jacrev(_boozer_residual_vector_composed)(x, **kwargs)
    return r, J
```

Caveat: the first-pass F2 finding already noted that `r` is recomputed inside `jacfwd`. Calling residual once outside (as above) makes that explicit; the user still pays one redundant evaluation in JIT compilation because `jacfwd` retraces.

### Untested edge cases

- `n_res < n_dofs` (small fixture, big surface basis) — no regression test.
- A `mpol=ntor=8` fixture with `nphi=ntheta=4` is highly unphysical but valid input; no guard.

---

## D4 — MEDIUM-PERF: `boozer_residual_jacobian_composed` traces residual twice

**File**: `src/simsopt/geo/boozer_residual_jax.py:738-740`

```python
r = _boozer_residual_vector_composed(x, **kwargs)
J = jax.jacfwd(_boozer_residual_vector_composed)(x, **kwargs)
return r, J
```

`jacfwd(f)(x)` internally evaluates `f(x)` to seed the forward pass primals. So the residual is computed twice in the eager trace:

1. line 738 — primal residual.
2. inside `jacfwd` (line 739) — primals seeded by re-invoking `_boozer_residual_vector_composed`.

After XLA tracing and CSE (common subexpression elimination), the compiled program should merge them — but only if **both calls use identical kwargs and tracer identity**. The `**kwargs` splat creates a new dict each call, but the leaves should be identical Python objects (no copies), so CSE should fire. **However**, JAX 0.10.0 does not guarantee perfect CSE across `jacfwd` boundaries when the function uses lazily-imported sub-functions (here `_get_grouped_biot_savart()` and `_get_surface_xyzfourier_fns()` re-import on each call).

### Verification needed

Run a focused test:

```python
# inside test_boozer_derivatives_jax.py
def test_jacobian_no_double_tracing(self):
    """Compile-once; trace count for residual pipeline should be 1."""
    import jax
    # ... call boozer_residual_jacobian_composed under a tracing counter ...
    # Assert _composed_pipeline traces exactly once.
```

### Recommendation

Use `jax.linearize` or extract primals from the jacfwd result if the JAX version supports it:

```python
def boozer_residual_jacobian_composed(x, **kwargs):
    # jax.jacfwd internally constructs primals; we can use linearize + vmap to
    # get the same result for one fewer trace.
    primal, jvp_fn = jax.linearize(
        lambda y: _boozer_residual_vector_composed(y, **kwargs), x
    )
    eye = jnp.eye(x.shape[0], dtype=x.dtype)
    J = jax.vmap(jvp_fn)(eye).T  # or equivalent
    return primal, J
```

This is a perf optimization, not a correctness fix.

---

## D5 — MEDIUM: Sibling API defaults disagree on `weight_inv_modB`

**File**: `src/simsopt/geo/boozer_residual_jax.py:293, 752`

| Function | Default `weight_inv_modB` |
|----------|----------------------------|
| `boozer_residual_scalar` | `True` (line 123) |
| `boozer_residual_vector` | `True` (line 293) |
| `boozer_penalty_composed` | `True` (line 611) |
| `_boozer_residual_vector_composed` | `False` (line 688) |
| `boozer_residual_coil_vjp` | `False` (line 752) |

The composed scalar pipeline defaults to `True`, but the residual vector / coil VJP path defaults to `False`. The CPU oracle (`surfaceobjectives.boozer_surface_residual` at line 541) defaults to `False`.

### Why this matters

- **Production callers** in `boozersurface_jax.py` always pass `weight_inv_modB` explicitly, so this doesn't bite the LS solver. But a test or research script calling `boozer_residual_coil_vjp(...)` without specifying `weight_inv_modB` is currently doing **unweighted** adjoints — silently different from the scalar pipeline.

- The CPU oracle's default is `False` (line 541) but the wrapper in `boozersurface.py:727` flips that to `True`. A reviewer expecting "JAX defaults match the CPU oracle directly" will read the JAX `boozer_residual_scalar(weight_inv_modB=True)` and conclude (wrongly) that the CPU oracle was being called with `weight_inv_modB=True` by default — that's only true through the `boozer_penalty_constraints_vectorized` wrapper.

### Recommendation

Document the inversion explicitly in `boozer_residual_coil_vjp` docstring, **or** flip the default to `True` to match the production exercise path. (Defaulting to `True` is safer since the unweighted path scales with coil currents.)

### Untested edge case

- `boozer_residual_coil_vjp` adjoint test (test_coil_vjp_*) calls with `weight_inv_modB=False`. There is no `weight_inv_modB=True` adjoint test for the coil VJP. The first pass would not have caught this because the forward formula parity is fine — the gap is only visible under the chain `outer-adjoint -> coil_vjp` where the upstream adjoint vector was sized against the weighted residual but the coil_vjp evaluates the unweighted residual.

---

## D6 — MEDIUM: |B|=0 degeneracy is "documented but unhandled"

**File**: `src/simsopt/geo/boozer_residual_jax.py:85-87, 95-96`

```python
def _inverse_modB(B2):
    """Return ``1 / |B|``; degenerate zero-field inputs surface as non-finite."""
    return _explicit_rsqrt(B2)
```

`_explicit_rsqrt` (in `_math_utils.py:146-156`) returns `1 / sqrt(x)`, which is `+inf` at `x=0`. The custom JVP rule is `-0.5 / (x * sqrt(x))`, which is `+inf` at `x=0`.

### Edge case behavior

`test_weighted_zero_field_is_nonfinite` (test_boozer_residual_jax.py:262) confirms that the scalar/vector outputs are non-finite at |B|=0. **The gradient at |B|=0 is also non-finite**, but no test verifies this. A solve that visits a |B|=0 quadrature point (e.g., a deep magnetic null inside the surface) will produce `NaN` gradients that propagate through the optimizer.

### Recommendation

Either:
1. Mask the residual at |B|=0 points (set residual to 0 there, with explicit warning) — this is what some CPU codes do; or
2. Document explicitly that `weight_inv_modB=True` requires `|B| > 0` at all quadrature points and raise at adapter boundary if violated.

Option 2 is consistent with the docstring intent; the explicit guard belongs in `BoozerSurfaceJAX._validate_state()` or equivalent.

### Untested edge cases

- |B| near (but not at) zero — gradient magnitudes near floating-point overflow.
- Surface inside a magnetic null (axis-search use case).
- A quadrature point where only one component of B is zero but |B|² is non-zero — fine but worth a test.

---

## D7 — LOW: Float32 inputs silently promoted to float64

**File**: `src/simsopt/jax_core/_math_utils.py:61-69`, applied at `boozer_residual_jax.py:144-147`

```python
def as_runtime_array(value, *, dtype, reference):
    # ``reference`` is accepted for call-site symmetry with tracer-aware APIs;
    # conversion is device-uniform and does not branch on it.
    del reference
    return as_jax_array(value, dtype=dtype)


def as_runtime_float64(value, *, reference):
    return as_runtime_array(value, dtype=jnp.float64, reference=reference)
```

The `reference` argument is **discarded** (`del reference`). The output is unconditionally promoted to `float64`. Callsites pattern:

```python
G = _as_runtime_float64(G, reference=B)
iota = _as_runtime_float64(iota, reference=B)
```

### Pitfall

If a user constructs `B` as `float32` (e.g., for GPU memory bandwidth experiments), `B.dtype` is `float32` but `G` and `iota` are silently promoted to `float64`. The product `G * B` then promotes the entire pipeline to `float64`. The user wanted `float32` throughout; the runtime gives `float64` without warning.

Worse, the input `B` is **not** demoted/promoted by `_as_runtime_float64` (it's never passed through `_as_*`), so the residual `G * B - B2[..., None] * tang` operates with mixed dtypes — the result is `float64` and the user's float32 intent is silently discarded.

### Why "low" severity

CLAUDE.md explicitly states "JAX 0.10.0, ... NumPy 2.x" and the module is float64-only by design. The strict CPU/JAX byte-identity gate forbids float32. So this is a "user misuse won't be caught early" issue, not a correctness defect under the supported configuration.

### Recommendation

Add an assertion in `boozer_residual_scalar` (and its callers) at entry:

```python
def boozer_residual_scalar(G, iota, B, xphi, xtheta, ...):
    if B.dtype != jnp.float64:
        raise TypeError(
            f"boozer_residual_scalar requires float64 B; got {B.dtype}. "
            "The strict parity contract is float64-only (see CLAUDE.md)."
        )
    ...
```

This makes the contract enforceable and avoids the silent promotion footgun.

---

## D8 — LOW: C++ kernel has no UB patterns; one SIMD invariant relies on allocator padding

**File**: `src/simsoptpp/boozerresidual_impl.h`

### Check 1 — `pragma omp` races

`grep -n "pragma omp" boozerresidual_impl.h` returns no matches. The kernel is **single-threaded**. No race window like the previously-fixed `mod_B_squared` race in `integral_BdotN.cpp`.

### Check 2 — Missing `if` braces

`grep -nE "if\s*\(.*\)\s*[a-zA-Z]"` returns no problematic patterns. All `if(...)` constructs use either `{ ... }` blocks or terminate at `;`. No naked-statement bug like the surfacerzfourier.cpp `ANGLE_RECOMPUTE` pattern.

### Check 3 — Signed/unsigned mixing

The loop indices are `int i, j, m, n` and the bound is `ndofs` which is `size_t`. The comparison `m < ndofs` (int < size_t) is a signed/unsigned comparison; GCC/clang both promote the int to unsigned, which is safe for `m >= 0`. Standard C++ behavior; not a bug.

### Check 4 — Uninitialized variables

`AlignedPaddedVec(ndofs+2, 0)` initializes all `ndofs+2` elements to 0. The over-allocation (padding for SIMD alignment) is uninitialized but is **never read** as a meaningful value — the SIMD load at `&drtilij0[m]` is followed by an explicit `jjlimit = min(simd_size, ndofs+2-m)` and writes only `jj < jjlimit` lanes. **However**, in the line 213 expression `simd_t d2res_mn = drtilij0_dm * drtilij0_dn + ...`, the full simd vector is multiplied. Padding lanes contain garbage (or zero if the allocator zero-fills, which it does NOT guarantee). The result is harmless because:

- The product of garbage with garbage produces NaN at worst; only the first `jjlimit` lanes are then written to `d2res(m+jj, n)`.

But this is a "harmless because we mask the output" pattern — if a future change forgets the masking, garbage NaNs leak into the Hessian. **Add a code comment documenting the invariant.**

### Check 5 — Dead code paths

The SIMD branch (line 13-343) and the non-SIMD branch (line 346-558) duplicate the kernel. They are not exercised against each other in any test fixture (no `USE_XSIMD=0` build coverage). A regression in the non-SIMD branch would not be caught by any CI. **Add a single CPU/non-SIMD parity probe.**

### Recommendation

- Add comment at line 205 about the SIMD load over-read invariant.
- Add a CI probe that runs at least one boozer_residual test with `-DUSE_XSIMD=0` to cover the non-SIMD branch.

---

## D9 — Physics convention check (informational)

The Boozer residual takes the form

```
r = G·B − |B|² · (∂γ/∂φ + ι·∂γ/∂θ)
```

The JAX implementation at `_boozer_weighted_residual` (line 91-93):
```python
tang = xphi + iota * xtheta
residual = G * B - B2[..., None] * tang
```

uses `+ iota * xtheta`. The CPU oracle at `surfaceobjectives.py:599`:
```python
tang = xphi + iota * xtheta
```

uses `+ iota * xtheta`. Identical sign.

The Boozer-coordinate convention is `m·θ − n·ζ` in the helical mode definition, but the **residual** is independent of that mode convention because it operates on `(∂γ/∂φ, ∂γ/∂θ)` directly. The sign convention in the residual is fixed by `B_BS · ∇φ = G_curl` for vacuum fields, where φ is the geometric toroidal angle. **No convention drift between JAX and CPU.**

The PEST vs Boozer distinction enters at the surface-from-DOFs stage (whether the angles `θ, ζ` are the straight-field-line or Boozer angles), not in the residual definition. The JAX module is **angle-agnostic** at this layer; correctness inherits from `surface_fourier_jax.py`.

---

## Test Fixture Inventory (Gaps)

### Existing fixtures (test_boozer_residual_jax.py + test_boozer_derivatives_jax.py)

- `_make_synthetic_data(nphi=10, ntheta=12)` — random B/xphi/xtheta.
- `_make_torus_dofs(mpol=1, ntor=0, nfp=1, R=1.0, r=0.1)` — circular torus, non-stellsym.
- `_make_torus_dofs_stellsym(mpol=1, ntor=0, nfp=1)` — circular torus, stellsym.
- `_make_coil_data(ncoils=3, nquad=32)` — three circular coils, 1e5 A each.
- `_make_near_floor_data(nphi=48, ntheta=48, residual_scale=1e-12)` — near-floor stress.
- `_make_scalar_dynamic_range_data` — 1e8/1.0 amplitude mix for compensated-sum tests.

### Missing fixtures (recommended additions)

| ID | Fixture | Tests it would unlock |
|----|---------|------------------------|
| T1 | `_make_iota_zero_data` | iota=0 exact gradient/Hessian check (avoids `xphi + 0·xtheta` degeneracy) |
| T2 | `_make_G_zero_data` | G=0 (zero-current limit); residual should reduce to `-|B|² · tang`; gradient w.r.t. iota must still be finite |
| T3 | `_make_zero_currents_coil_data` | All currents zero → `compute_G_from_currents` returns 0; tests the `optimize_G=False` auto-G path correctness |
| T4 | `_make_multi_helicity_dofs(mpol=4, ntor=4)` | Non-trivial m·n cross-coupling; tests jacfwd memory at moderate scale |
| T5 | `_make_near_axis_dofs(R=0.1, r=0.5)` | Surface near the magnetic axis where `|tang|` is small; weight_inv_modB=True regime |
| T6 | `_make_self_intersecting_dofs` | DOFs that produce a self-intersecting surface (negative Jacobian regions); should fail gracefully or produce NaN gradient (currently undefined behavior) |
| T7 | `_make_float32_input_data` | Confirm strict typing — should either work or raise informatively |
| T8 | `_make_mismatched_decision_vector` | x of length 0, 1, 2 with both `optimize_G=True/False`; should raise informatively |
| T9 | `_make_large_currents_data` | currents ~ 1e7 A; G ~ 0.04 Tm; magnitude of residual scales correctly |
| T10 | `_make_grouped_coil_data` | Two groups of coils with different `nquad` (15 vs 128, mirroring the TF+banana mixed-quadrature support in CLAUDE.md); tests `_unpack_decision_vector` with multi-group `coil_arrays` |

### Suggested new tests

1. **`test_decision_vector_too_short_raises`** — sanity check D1.
   ```python
   def test_optimize_G_too_short_raises():
       from simsopt.geo.boozer_residual_jax import _split_decision_vector
       with pytest.raises(ValueError):
           _split_decision_vector(jnp.array([1.0]), optimize_G=True)
   ```

2. **`test_G_from_currents_consistency`** — assert duplicate inline G formula matches `compute_G_from_currents`.
   ```python
   def test_inline_G_matches_compute_G_from_currents():
       from simsopt.geo.boozer_residual_jax import _unpack_decision_vector
       from simsopt.geo.label_constraints_jax import compute_G_from_currents
       currents = jnp.array([1.5e6, -2.1e6, 3.0e5])
       coil_arrays = [(jnp.zeros((3, 4, 3)), jnp.zeros((3, 4, 3)), currents)]
       x = jnp.zeros(1)  # 1 surface DOF + 1 iota
       _, _, G_inline = _unpack_decision_vector(x, coil_arrays, optimize_G=False)
       G_ssot = compute_G_from_currents(currents)
       np.testing.assert_allclose(float(G_inline), float(G_ssot), rtol=0, atol=0)
   ```

3. **`test_jacfwd_jacrev_agree`** — guards against autodiff regressions in the composed Jacobian.
   ```python
   def test_jacfwd_jacrev_agree(self):
       _, J_fwd = boozer_residual_jacobian_composed(self.x, **self.kwargs)
       J_rev = jax.jacrev(_boozer_residual_vector_composed)(self.x, **self.kwargs)
       np.testing.assert_allclose(np.asarray(J_fwd), np.asarray(J_rev), rtol=1e-12, atol=1e-14)
   ```

4. **`test_coil_vjp_weighted`** — covers the `weight_inv_modB=True` path on the coil VJP that is currently uncovered (D5).

5. **`test_residual_gradient_at_zero_B_is_nonfinite`** — explicit guard for D6.

6. **`test_jacobian_with_zero_currents`** — D2 edge case (T3).

7. **`test_jacobian_with_iota_zero`** — T1 fixture exercised through `boozer_residual_jacobian_composed`.

8. **`test_grouped_coil_arrays_grad_unchanged`** — covers `_unpack_decision_vector` with `len(coil_arrays) > 1`.

---

## Comparison with CPU Oracle (API Surface)

CPU `boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False)`:
- accepts `derivatives in {0, 1, 2}`
- `G=None` triggers auto-G from currents (per `surfaceobjectives.py:577-584`)
- defaults `weight_inv_modB=False`
- returns residual vector + Jacobian + Hessian (depending on `derivatives`)

JAX M3 layer offers `(scalar, vector, jacobian, vjp)` as **separate functions**:
- `boozer_penalty_composed` — scalar
- `_boozer_residual_vector_composed` — vector
- `boozer_residual_jacobian_composed` — vector + Jacobian
- `boozer_residual_coil_vjp` — coil-VJP (different shape than CPU)
- **No `boozer_residual_hessian_composed`**

### Gap

The CPU oracle returns the full residual Hessian (`d²r/dx²`) at `derivatives=2`. The JAX module offers `jax.hessian(boozer_penalty_composed)` for the **scalar** Hessian, but not the **vector** Hessian (i.e., the per-component d²r/dx²). The vector Hessian is needed by the C++ kernel `sopp.boozer_residual_ds2` for Newton's method on the residual.

**Status**: Acceptable gap. The JAX path uses the **scalar** objective for L-BFGS / Newton-on-J and the **vector + Jacobian** for least-squares — both are covered. The vector Hessian is only consumed by the CPU exact-mode Newton solver, which has its own JAX equivalent (`newton_exact` in `optimizer_jax.py`). Verified by reading `boozersurface_jax.py:2020` and adjoint runtime state code path.

### Default mismatch

CPU default `weight_inv_modB=False`; JAX defaults split between `True` (scalar, vector, composed scalar) and `False` (coil VJP, residual vector composed). See D5. **Recommended fix**: align all to `True` matching production exercise; document the inversion if not.

---

## Summary of Recommended Code Changes

1. **D1 fix** (HIGH): add `surface_size >= 0` validation in `_split_decision_vector`.
2. **D2 fix** (HIGH): replace inline G-from-currents in `_unpack_decision_vector` with a (lazy) import of `compute_G_from_currents`.
3. **D3 fix** (MEDIUM): add `n_res < n_dofs ⇒ jacrev` heuristic to `boozer_residual_jacobian_composed`.
4. **D4 fix** (MEDIUM-PERF): consider `jax.linearize`-based Jacobian to avoid double residual trace.
5. **D5 fix** (MEDIUM): align defaults of `boozer_residual_coil_vjp` and `_boozer_residual_vector_composed` to `weight_inv_modB=True`, or add explicit docstring inversion notice.
6. **D6 fix** (MEDIUM): document |B|=0 gradient non-finiteness and add adapter-level guard.
7. **D7 fix** (LOW): assert `B.dtype == jnp.float64` at entry to scalar/vector residual functions.
8. **D8 fix** (LOW): add code comment in `boozerresidual_impl.h:205` documenting SIMD over-read invariant; add CI probe for `USE_XSIMD=0` branch.

## Summary of Recommended New Tests

| ID | Test | Severity Mapping |
|----|------|------------------|
| T1 | `test_iota_zero_gradient_finite` | T1 fixture |
| T2 | `test_G_zero_residual_form` | D2 + T2 |
| T3 | `test_zero_currents_optimize_G_false_yields_zero_G` | D2 + T3 |
| T4 | `test_jacobian_at_mpol_ntor_4` (mid-scale memory probe) | D3 + T4 |
| T5 | `test_jacfwd_jacrev_agree` | D3, D4 |
| T6 | `test_coil_vjp_weighted_inv_modB_true` | D5 |
| T7 | `test_gradient_at_zero_B_is_nonfinite` | D6 |
| T8 | `test_decision_vector_validation` | D1 |
| T9 | `test_inline_G_matches_SSOT` | D2 |
| T10 | `test_float32_input_raises_informatively` | D7 |
| T11 | `test_grouped_coil_arrays_consistent` | D2 cross-group |
| T12 | `test_non_simd_cpu_path_parity` (build-flag-conditional) | D8 |

None of these findings represent a silent wrong-answer path under default production usage (verified by tracing `boozersurface_jax.py` callers). All findings are either contract drift, missing input validation, performance heuristics, or fixture-coverage gaps. The forward-formula parity established by the first pass remains valid; the deeper review identifies hardening opportunities rather than correctness defects.
