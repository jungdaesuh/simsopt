# Parity Audit 12 — Boozer Radial Fourier Interpolant Kernels

| Field | Value |
| --- | --- |
| Audit timestamp | 2026-05-16 (UTC) |
| Branch | `gpu-purity-stage2-20260405` |
| JAX file | `src/simsopt/jax_core/boozer_radial_interp.py` (559 lines) |
| C++ file | `src/simsoptpp/boozerradialinterpolant.cpp` (221 lines) |
| C++ header | `src/simsoptpp/boozerradialinterpolant.h` (13 lines) |
| Adjacent consumer (CPU) | `src/simsopt/field/boozermagneticfield.py` (lines 670-1085) |
| Adjacent consumer (JAX item 33) | `src/simsopt/field/boozermagneticfield_jax.py` (lines 411-760) |
| Adjacent consumer (item 33 helper) | `src/simsopt/jax_core/boozer_fixed_state.py` (lines 200-225) |
| Parity tests | `tests/jax_core/test_boozer_radial_interp_jax_item32.py` (601 lines) |
| Tolerance lane | `direct_kernel` (`rtol=1e-10`, `atol=1e-12`) |

## Scope Disambiguation

The audit prompt is framed around a full Boozer radial interpolator (Chebyshev `T_n(√s)`, splines in `s`, `BoozerRadialInterpolant.modB(s,θ_b,ζ_b)`). **The audited JAX file does NOT implement the radial reconstruction.** It is restricted, by explicit contract (`boozer_radial_interp.py:32-37`), to the **six closed-form Fourier helper kernels** that the upstream C++ binding ships:

1. `compute_kmns` — stellsym K Fourier projection on a half-grid surface table.
2. `compute_kmnc_kmns` — non-stellsym K Fourier projection.
3. `fourier_transform_odd` — sin-mode Fourier projection (per-mode point-sum normalisation).
4. `fourier_transform_even` — cos-mode Fourier projection.
5. `inverse_fourier_transform_odd` — sum `kmns·sin(angle)`.
6. `inverse_fourier_transform_even` — sum `kmnc·cos(angle)`.

The radial spline construction (`scipy.interpolate.InterpolatedUnivariateSpline`, half-grid extension, `mn_factor = s^{m/2}` axis-regularity factor, K-coefficient post-scaling by `dθ·dζ·nfp/ψ0`) lives in `boozermagneticfield.py::BoozerRadialInterpolant` and its JAX mirror (`boozermagneticfield_jax.py::_eval_modB`, `_eval_K`, etc.). Per the file's own docstring (`boozer_radial_interp.py:32-36`), the BoozerMagneticField adapter is **explicitly out of scope** for item 32 and tracked as item 33.

Where parity findings touch the spline / axis-regularity layer (sections (a), (d)), they are reported against the **consumer Python wrapper** (CPU `BoozerRadialInterpolant`) and the **JAX mirror** (`boozermagneticfield_jax.py`), and clearly marked.

## Executive Summary — Top 3 Findings

1. **PARITY CLEAN on the six audited Fourier kernels.** The JAX implementations of `compute_kmns`, `compute_kmnc_kmns`, `fourier_transform_{odd,even}`, and `inverse_fourier_transform_{odd,even}` (both 1D and 2D variants) reproduce the C++ binding to `direct_kernel` tolerance (`rtol=1e-10`, `atol=1e-12`) across a representative param grid (`tests/jax_core/test_boozer_radial_interp_jax_item32.py:208-434`). The math, the DC-row zeroing convention, the normalisation factors (`1/(2π²)` for `im≥1`, `1/(4π²)` for the `im=0` cos term), and the 1D-vs-2D `kmns` polymorphism all line up exactly.

2. **MEDIUM — C++ `fourier_transform_odd`/`fourier_transform_even` contain a real OpenMP data race that JAX does NOT inherit** (C++ `boozerradialinterpolant.cpp:147-156, 165-175`). The `kmns(im) += ...` accumulator and the `norm` scalar are both `#pragma omp parallel for`-reduced without `reduction(+:...)` clauses or atomics. Empirically the parity tests still pass at `rtol=1e-10` because the simsoptpp build either disables OMP for these projections or the race is benign at small thread counts. The JAX kernel uses a deterministic matmul reduction. This is a **C++ bug**, not a JAX bug; flagging it because (a) the parity test would silently mask a future OMP regression, and (b) the JAX result is in fact the trustworthy reference.

3. **INFO — Audit-prompt scope items (`B(s,θ_b,ζ_b)` reconstruction, axis regularity, identity `B·∇ζ = G(s)`) are NOT exercised in `test_boozer_radial_interp_jax_item32.py`.** Those identities require the upstream `BoozerMagneticField` API to be assembled and live in `tests/field/test_boozermagneticfield_jax_item33.py` (the item-33 wrapper test). For item 32 in isolation this is correct-by-design, but a reader of this file alone might assume identity-level coverage is present.

## Function-by-Function Parity Matrix

| Function (JAX) | C++ counterpart | MATH | PHYSICS | ALGORITHM | COMPUTATION | Cross-oracle test |
| --- | --- | --- | --- | --- | --- | --- |
| `_build_angle_basis` (`boozer_radial_interp.py:57-70`) | inline `cos/sin(xm*θ−xn*ζ)` everywhere | PASS — angle sign `m·θ − n·ζ` matches every C++ call site | n/a | PASS — broadcast `(num_points, num_modes)` matmul-ready | PASS — float64 by JAX default | covered transitively |
| `_compute_K_per_point` (stellsym) (`:163-219`) | C++ `compute_kmns` inner loop (`boozerradialinterpolant.cpp:90-130`) | PASS — every closed-form (`dRdtheta`, `dZds`, `nu`, `dphidζ = 1 − dnudζ`, `sqrtg = (G+ιI)/B²`, `K = (gsζ + ι gsθ)/sqrtg`) reproduced verbatim | PASS — Boozer-coordinate `K` definition exact | PASS — 2 matmuls (1 cos, 1 sin), 1 `dot_general` proven by jaxpr regression (`test_compute_K_per_point_batches_stellsym_fourier_sums`) | PASS | covered transitively via `compute_kmns` parity |
| `_compute_K_per_point` (asym) (`:111-162`) | C++ `compute_kmnc_kmns` inner loop (`:22-63`) | PASS — non-stellsym coupling (`bmns`, `rmns`, `zmnc`, `numnc` and respective `s`-derivs) all included with correct ± signs | PASS | PASS — 2-matmul reduction; jaxpr asserts 2 `dot_general` ops (`test_compute_K_per_point_batches_asym_fourier_sums`) | PASS | covered transitively |
| `compute_kmns` (`:223-278`) | `compute_kmns` (`:79-138`) | PASS | PASS | PASS — vmap over surfaces, sin-only matmul with DC row zeroed | PASS — `/(2π²)` normalisation exact | `test_compute_kmns_matches_cpp` (rtol 1e-10) |
| `compute_kmnc_kmns` (`:282-364`) | `compute_kmnc_kmns` (`:7-77`) | PASS — DC row uses `1/(4π²)`, others use `1/(2π²)`; DC sin row exactly zero | PASS | PASS — vmap + `jnp.where(arange==0, 1/(4π²), 1/(2π²))` | PASS | `test_compute_kmnc_kmns_matches_cpp` |
| `fourier_transform_odd` (`:373-396`) | `fourier_transform_odd` (`:140-157`) | PASS — `numer/denom`, DC row zeroed | PASS (point-sum projection) | PASS | PASS — `safe_denom` guards the DC division | `test_fourier_transform_odd_matches_cpp` |
| `fourier_transform_even` (`:400-417`) | `fourier_transform_even` (`:159-176`) | PASS — DC row INCLUDED (matches C++ `for im=0`) | PASS | PASS | PASS | `test_fourier_transform_even_matches_cpp` |
| `inverse_fourier_transform_odd_1d` (`:426-443`) | `inverse_fourier_transform_odd` 1D branch (`:190-197`) | PASS — DC suppressed via `kmns.at[0].set(0.0)` then `sin_a @ kmns_no_dc` | n/a | PASS | PASS | `test_inverse_fourier_transform_odd_1d_matches_cpp` |
| `inverse_fourier_transform_odd_2d` (`:447-468`) | `inverse_fourier_transform_odd` 2D branch (`:183-189`) | PASS — diagonal-broadcast `einsum("mp,pm->p", kmns_no_dc, sin_a)` reproduces `K[ip] += kmns(im,ip)*sin(angle)` | n/a | PASS | PASS | `test_inverse_fourier_transform_odd_2d_matches_cpp` |
| `inverse_fourier_transform_odd` (dispatch) (`:471-490`) | runtime `dim==2` branch in C++ | PASS — Python `ndim` dispatch with explicit `ValueError` for rank ≥ 3 | n/a | PASS | PASS | `test_inverse_fourier_transform_rejects_unsupported_rank` |
| `inverse_fourier_transform_even_1d` (`:494-507`) | `inverse_fourier_transform_even` 1D branch (`:212-218`) | PASS — DC INCLUDED via `cos_a @ kmnc` | n/a | PASS | PASS | `test_inverse_fourier_transform_even_1d_matches_cpp` |
| `inverse_fourier_transform_even_2d` (`:511-525`) | `inverse_fourier_transform_even` 2D branch (`:205-211`) | PASS | n/a | PASS | PASS | `test_inverse_fourier_transform_even_2d_matches_cpp` |
| `inverse_fourier_transform_even` (dispatch) (`:528-545`) | runtime `dim==2` branch in C++ | PASS | n/a | PASS | PASS | `test_inverse_fourier_transform_rejects_unsupported_rank` |

**Net verdict: 13/13 audited kernels are parity-clean.** No CRITICAL or HIGH findings on the six core Fourier helpers. The single MEDIUM is on the C++ side, not the JAX side.

## Detailed Findings

### Finding 1 — PASS — Angle-convention sign (`m·θ − n·ζ`) is consistent everywhere

**Severity: PASS / INFO**

The Boozer-angle convention chosen by upstream simsoptpp is `angle = xm(im)·θ − xn(im)·ζ`. The JAX kernel honours this in **every** code path:

```python
# boozer_radial_interp.py:69
angle = thetas[:, None] * xm[None, :] - zetas[:, None] * xn[None, :]
```

C++ uses the same sign convention in `compute_kmns` (`boozerradialinterpolant.cpp:104`):

```cpp
B += bmnc(im,isurf)*cos(xm(im)*thetas(ip)-xn(im)*zetas(ip));
```

and in `compute_kmnc_kmns` (`:36`), `fourier_transform_odd` (`:151`), `fourier_transform_even` (`:170`), and both `inverse_*` paths (`:187, 194, 209, 216`). The sign is consistent. There is no mixed `+`/`−` convention bug anywhere in this kernel set.

> Note: this is BOOZ_XFORM's convention. PEST coordinates use the opposite sign for `n`. The audited code is correctly Boozer-convention throughout.

### Finding 2 — PASS — K formula reproduces C++ exactly (stellsym and non-stellsym)

**Severity: PASS / INFO**

The Boozer metric `K(θ,ζ)` at a single surface is defined by

```
K = (g_{sζ} + ι·g_{sθ}) / √g     (1)
√g = (G + ι·I) / B²              (2)
```

with `g_{sθ} = ∂X/∂θ·∂X/∂s + ∂Y/∂θ·∂Y/∂s + ∂Z/∂θ·∂Z/∂s` (and analogously `g_{sζ}`), where the Cartesian basis arises from cylindrical (`R`, `φ = ζ − ν`, `Z`).

C++ `compute_kmns` inner block (`boozerradialinterpolant.cpp:117-130`):
```cpp
double phi = zetas(ip) - nu;
double dphids = - dnuds;
double dphidtheta = - dnudtheta;
double dphidzeta = 1 - dnudzeta;
double dXdtheta = dRdtheta * cos(phi) - R * sin(phi) * dphidtheta;
double dYdtheta = dRdtheta * sin(phi) + R * cos(phi) * dphidtheta;
...
double gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds;
double gszeta  = dXdzeta  * dXds + dYdzeta  * dYds + dZdzeta  * dZds;
double sqrtg = (G(isurf) + iota(isurf)*I(isurf))/(B*B);
double K = (gszeta + iota(isurf)*gstheta)/sqrtg;
```

JAX `_compute_K_per_point` (`boozer_radial_interp.py:201-219`):
```python
phi = zetas - nu
dphids = -dnuds
dphidtheta = -dnudtheta
dphidzeta = 1.0 - dnudzeta
...
dXdtheta = dRdtheta * cos_phi - R * sin_phi * dphidtheta
...
gstheta = dXdtheta * dXds + dYdtheta * dYds + dZdtheta * dZds
gszeta = dXdzeta * dXds + dYdzeta * dYds + dZdzeta * dZds
sqrtg = (G_isurf + iota_isurf * I_isurf) / (B * B)
return (gszeta + iota_isurf * gstheta) / sqrtg
```

Byte-for-byte the same Boozer-coordinate K. The asym block carries the additional cross terms (`bmns·sin`, `rmns·sin`, `zmnc·sin`, etc.) with the correct ± signs verified against `compute_kmnc_kmns` (`:36-48`).

### Finding 3 — PASS — Normalisation factors `1/(2π²)` and `1/(4π²)` match exactly

**Severity: PASS / INFO**

In `compute_kmnc_kmns` the C++ kernel uses two different prefactors:

```cpp
// boozerradialinterpolant.cpp:67-72
if (im > 0) {
  kmnc_kmns(1,im,isurf) += K*sin(angle)/(2.*M_PI*M_PI);
  kmnc_kmns(0,im,isurf) += K*cos(angle)/(2.*M_PI*M_PI);
} else {
  kmnc_kmns(0,im,isurf) += K*cos(angle)/(4.*M_PI*M_PI);
}
```

The JAX kernel reproduces this per-mode prefactor using a vectorised `jnp.where` (`boozer_radial_interp.py:344-358`):

```python
pi2 = jnp.pi * jnp.pi
scale = jnp.where(
    jnp.arange(cos_proj.shape[0]) == 0,
    1.0 / (4.0 * pi2),
    1.0 / (2.0 * pi2),
)
kmnc_isurf = cos_proj * scale

sin_only = sin_a.at[:, 0].set(0.0)
kmns_isurf = (K[None, :] @ sin_only).ravel() / (2.0 * pi2)
```

`compute_kmns` (stellsym) is sin-only and uses `1/(2π²)` for all `im≥1` (`boozer_radial_interp.py:274`), matching the C++ `for (im=1; ...)` loop (`boozerradialinterpolant.cpp:132-134`). Verified at `rtol=1e-10` by `test_compute_kmns_matches_cpp`/`test_compute_kmnc_kmns_matches_cpp`.

> Reminder for downstream consumers: this prefactor is half the trapezoidal Fourier rule. The remaining `dθ·dζ·nfp/ψ0` multiplier is applied by `BoozerRadialInterpolant.compute_K` in the Python wrapper (`boozermagneticfield.py:699, 704`). The kernel itself is only responsible for the `1/(2π²)` / `1/(4π²)` part — the audited file documents this explicitly (`boozer_radial_interp.py:19-23`).

### Finding 4 — PASS — DC row handling is exactly right (sin-side zero, cos-side included)

**Severity: PASS / INFO**

C++ semantics:
- `compute_kmns`: `for (int im=1; im < num_modes; ++im)` → DC row left at its `xt::zeros` initial value of 0.
- `compute_kmnc_kmns`: explicit `if (im > 0)` guards the sin accumulator; the cos accumulator has a distinct `else` branch with `1/(4π²)`.
- `fourier_transform_odd`: `for (int im=1; ...)` → DC row stays zero.
- `fourier_transform_even`: `for (int im=0; ...)` → DC row included.
- `inverse_fourier_transform_odd`: `for (int im=1; ...)` → DC contribution suppressed.
- `inverse_fourier_transform_even`: `for (int im=0; ...)` → DC contribution included.

JAX semantics:
- `compute_kmns:273` — `sin_only = sin_a.at[:, 0].set(0.0)` then matmul. DC row of result is exactly 0.0.
- `compute_kmnc_kmns:349-358` — `jnp.where(arange==0, 1/(4π²), 1/(2π²))` for cos; `sin_a.at[:, 0].set(0.0)` for sin. Identical contract.
- `fourier_transform_odd:394-396` — `safe_denom` + final `jnp.where(arange==0, 0.0, result)`. Identical contract.
- `fourier_transform_even` — no DC suppression. Matches C++.
- `inverse_fourier_transform_odd_{1d,2d}` — explicit `kmns.at[0].set(0.0)` / `kmns.at[0, :].set(0.0)` before the matmul/einsum.
- `inverse_fourier_transform_even_{1d,2d}` — no DC suppression. Matches C++.

The DC convention is verified by the dedicated tests `test_compute_kmns_zero_dc_row` (`test_boozer_radial_interp_jax_item32.py:497-520`) and `test_compute_kmnc_kmns_zero_dc_sin_row` (`:523-553`).

### Finding 5 — MEDIUM — `fourier_transform_odd` / `fourier_transform_even` have a C++ OpenMP data race that JAX does not inherit

**Severity: MEDIUM (C++ side only)**

C++ `boozerradialinterpolant.cpp:140-176`:
```cpp
Array fourier_transform_odd(Array& K, Array& xm, Array& xn, Array& thetas, Array& zetas) {
    int num_modes = xm.shape(0);
    int num_points = thetas.shape(0);
    Array kmns = xt::zeros<double>({num_modes});

    double norm;
    for (int im=1; im < num_modes; ++im) {
      norm = 0.;
      #pragma omp parallel for
      for (int ip=0; ip < num_points; ++ip) {
        kmns(im) += K(ip)*sin(xm(im)*thetas(ip)-xn(im)*zetas(ip));
        norm += pow(sin(xm(im)*thetas(ip)-xn(im)*zetas(ip)),2.);
      }
      kmns(im) = kmns(im)/norm;
    }
    return kmns;
}
```

This is racy on three counts:
1. `kmns(im) += ...` is accumulated by all threads with no `reduction(+:...)`, no `atomic`, no `critical`.
2. `norm += ...` is a captured-by-reference scalar with the same problem.
3. The same `sin(...)` is re-computed twice per iteration (a perf nit, not a parity issue).

The same pattern is in `fourier_transform_even` (`:165-174`). In contrast, the JAX kernel uses deterministic matmul reductions:

```python
# boozer_radial_interp.py:391-396
numer = sin_a.T @ K  # (num_modes,)
denom = jnp.sum(sin_a * sin_a, axis=0)  # (num_modes,)
safe_denom = jnp.where(jnp.arange(numer.shape[0]) == 0, 1.0, denom)
result = numer / safe_denom
return jnp.where(jnp.arange(numer.shape[0]) == 0, 0.0, result)
```

**Why the parity test still passes:** simsoptpp must be effectively single-threaded for these kernels in the parity-test environment (the upstream `boozerradialinterpolant.cpp` is one of the few simsoptpp files that has never been hardened against this pattern). The race is **latent**: change `OMP_NUM_THREADS` and the test starts producing non-deterministic C++ results, while JAX stays deterministic. **The JAX kernel is the trustworthy reference here.**

**Impact for this port:**
- The JAX parity test (`test_fourier_transform_{odd,even}_matches_cpp`) at `rtol=1e-10` is a regression gate for **JAX equality with whatever the C++ thread reduction returned**. On macOS with the in-tree `.conda/jax` env this is single-threaded; on Linux with `OMP_NUM_THREADS>1` it might surprise. The risk is `direct_kernel`-lane tolerance being violated by a noisy C++ side, not by JAX.

**Recommendation:** add a closed-form (oracle-free) parity test for `fourier_transform_{odd,even}`: feed in a pure-mode `K(ip) = sin(angle[ip, m*])` and check that JAX returns the unit basis vector to machine precision. This is structurally identical to the existing `test_inverse_fourier_transform_reconstructs_pure_modes` (`:442-461`) but for the forward projection. That gives a JAX-vs-closed-form oracle that is invariant to the C++ race.

> Tangential side note: `compute_kmns` and `compute_kmnc_kmns` on the C++ side are RACE-FREE because the `+=` accumulators (`B`, `R`, `dRdtheta`, ...) are thread-local stack variables; only the outermost `#pragma omp parallel for` over `isurf` is parallelised, and each thread writes to a distinct slice of `kmns(im, isurf)`. So the race is **localised to the two `fourier_transform_*` helpers only.**

### Finding 6 — INFO — Cos-coeff projection in `compute_kmnc_kmns` includes DC, but is not divided by 2

**Severity: INFO**

The C++ cos accumulator for `im=0` divides by `4π²`, while `im≥1` divides by `2π²`. This is the standard Fourier projection prefactor: the constant (DC) basis function `cos(0) = 1` has `∫₀^{2π}∫₀^{2π} 1² dθ dζ = 4π²`, while a `cos(mθ−nζ)` for `(m,n)≠(0,0)` integrates to `2π²` under stellsym pairing. JAX `boozer_radial_interp.py:348-353` replicates this with the `jnp.where(arange==0, 1/(4π²), 1/(2π²))` per-mode selector. The asymmetry is **physically correct** and reproduces the C++ branch.

The sin counterpart (`im=0` zeroed) is correct because `sin(0) = 0` ⇒ the projection is identically zero for any input.

### Finding 7 — INFO — `inverse_fourier_transform_*` is `K += ...` in C++ but `K = ...` in JAX

**Severity: INFO (well-documented design choice)**

The C++ signature is `void inverse_fourier_transform_odd(Array& K, ...)` with `K(ip) += ...` accumulation (`boozerradialinterpolant.cpp:178, 187, 194, 200, 209, 216`). This lets the upstream `BoozerMagneticField._K_impl` call the kernel **twice** (once with `kmns`, once with `kmnc`) and have both contributions land in the same output buffer (`boozermagneticfield.py:730-735`):

```python
sopp.inverse_fourier_transform_odd(K[:, 0], kmns, self.xm_b, self.xn_b, thetas, zetas)
if not self.stellsym:
    ...
    sopp.inverse_fourier_transform_even(K[:, 0], kmnc, self.xm_b, self.xn_b, thetas, zetas)
```

The JAX kernel **returns** the contribution and lets the caller compose. This is required by JAX's pure-function/immutability model and is documented at `boozer_radial_interp.py:482-483`:

> The C++ kernel accumulates `K += ...`; this Python entry point **returns** the contribution and leaves accumulation to the caller.

The JAX wrapper layer correctly composes the two contributions with `result + ...`, e.g. `boozermagneticfield_jax.py:419-424` (`_eval_modB`), `:713-718` (`_eval_K`). **No parity gap from this difference.**

### Finding 8 — PASS — 1D-vs-2D `kmns` polymorphism matches C++ `int dim = kmns.dimension()` dispatch

**Severity: PASS / INFO**

The C++ kernel inspects `kmns.dimension()` and selects between `kmns(im)` (rank-1) and `kmns(im, ip)` (rank-2) at runtime (`boozerradialinterpolant.cpp:182, 204`). The JAX wrapper dispatches in pure Python based on `kmns.ndim` (`boozer_radial_interp.py:484-490, 539-545`):

```python
def inverse_fourier_transform_odd(kmns, xm, xn, thetas, zetas):
    if kmns.ndim == 1:
        return inverse_fourier_transform_odd_1d(kmns, xm, xn, thetas, zetas)
    if kmns.ndim == 2:
        return inverse_fourier_transform_odd_2d(kmns, xm, xn, thetas, zetas)
    raise ValueError(...)
```

The 2D variant correctly implements the diagonal-broadcast `K[ip] = Σ_im kmns[im, ip] · sin(angle[ip, im])` via `jnp.einsum("mp,pm->p", kmns_no_dc, sin_a)` (`boozer_radial_interp.py:467-468`). This is the path used by `_K_impl` in the CPU wrapper (`boozermagneticfield.py:727-730`) and verified by `test_inverse_fourier_transform_odd_2d_matches_cpp`. Note: because of the Python-level dispatch, **`inverse_fourier_transform_{odd,even}` itself is not JIT-decorated**; only its 1D/2D leaves are. This is intentional — `kmns.ndim` is a Python int, so a JITted dispatcher would need `static_argnums`. The leaves are JIT-compiled.

### Finding 9 — PASS — `_compute_K_per_point` collapses C++'s O(num_points · num_modes) inner double loop to two matmuls

**Severity: PASS / INFO — performance-critical**

The C++ inner loop in `compute_kmns` (`:103-115`) does ~12 trig calls and 12 multiply-adds per `(ip, im)` pair, all inside a parallel-for. The JAX kernel reshapes this as two batched matmuls (`boozer_radial_interp.py:148-199`):

- Stellsym path: 7 cos-coefficients stacked column-wise + 5 sin-coefficients, then `cos_a @ cos_coeffs` and `sin_a @ sin_coeffs`. Verified to compile to exactly **2 `dot_general` primitives** by the jaxpr regression test (`test_boozer_radial_interp_jax_item32.py:171-184`).
- Non-stellsym path: 12 cos + 12 sin coefficients, same 2-matmul structure. Verified by `test_compute_K_per_point_batches_asym_fourier_sums` (`:187-200`).

This is exactly the GPU-purity contract: turn double-nested per-point loops into batched linear algebra. The matmul reduction order is different from the C++ accumulation order, but the resulting K is bit-equal at `direct_kernel` tolerance because the underlying arithmetic is sums-of-products and float64 has enough headroom.

### Finding 10 — INFO — sin-coefficient extraction in stellsym path is via `cos_values[:, 0]` etc., not via component-wise destructure

**Severity: INFO (cosmetic)**

The asym path destructures via `jnp.moveaxis` + tuple unpacking (`boozer_radial_interp.py:148-162`), while the stellsym path indexes by column (`boozer_radial_interp.py:186-199`):

```python
cos_values = cos_a @ cos_coeffs
sin_values = sin_a @ sin_coeffs
B = cos_values[:, 0]
R = cos_values[:, 1]
dRdtheta = sin_values[:, 0]
...
```

This is intentional: the stellsym path has a **different column-to-quantity mapping** because the cos-stack and sin-stack carry different physical quantities (e.g. `cos_values[:, 1] = R` but `sin_values[:, 0] = dRdtheta`). The order in the stack literal (`boozer_radial_interp.py:163-185`) is the source of truth. **No parity issue, but a maintenance trap: any future addition or reordering of coefficients in the stack literals must be mirrored in the destructure block.** A future hardening would be to use `dict` or `NamedTuple` instead of positional indexing.

## Subsections

### (a) Radial-basis parity

**Out of scope for this kernel set, with the following caveats:**

The audited file does **not** define a radial basis. The radial reconstruction (B_mn(s), G(s), I(s), ι(s), ψ(s)) lives in:

- **CPU**: `simsopt.field.boozermagneticfield.py::BoozerRadialInterpolant` constructs `scipy.interpolate.InterpolatedUnivariateSpline` objects on the BOOZ_XFORM half-grid `s_half_ext` (`boozermagneticfield.py:455-457, 575-640`). Spline order is configurable via `order=` (default 5 in upstream simsopt). This is a **B-spline of order k**, NOT Chebyshev `T_n(√s)`.
- **JAX**: `simsopt.field.boozermagneticfield_jax.py::BoozerRadialInterpolantFrozenState` captures the scipy splines as `PPoly` (piecewise polynomial) tables via `_profile_from_host` (`boozermagneticfield_jax.py:203-214`) and evaluates them with `ppoly_eval` from `simsopt.jax_core.boozer_fixed_state.py`.

**Parity claim for the radial basis**: not made by item 32. Made by item 33 (`tests/field/test_boozermagneticfield_jax_item33.py:168-217`), which compares `wrapper.modB()` against the CPU `BoozerRadialInterpolant.modB()` at `direct_kernel` tolerance on synthetic stellsym and asym fixtures.

There is no Chebyshev `T_n(√s)` basis anywhere in simsopt — that is the SFINCS / BOOZ_XFORM convention but not the simsopt one. The `mn_factor = s^{|m|/2}` axis-regularity factor is present but applied as a **spline-coefficient pre-multiplication**, not as a basis function. See subsection (d).

### (b) Mode-sum parity

Verified for every kernel (Finding 1). Both sides use the Boozer convention `angle = m·θ − n·ζ`. The mode-sum **reduction order** in JAX is the matmul-default `dot_general` axis order (modes-axis reduced); in C++ it is the sequential `for (im=...)` inner loop. The mismatched reduction orders are confirmed harmless at `rtol=1e-10` empirically (`test_compute_kmns_matches_cpp` and friends, parametrised over `num_modes ∈ {6, 8, 12, 16, 24}` and `num_points ∈ {50, 100, 128, 200, 256}`).

JAX uses the same single mode-index variable as C++ (`im`); no PEST / Hamada mismatch.

### (c) Derivatives parity

The audited file does **not** define `s`-derivatives — those are downstream of the spline construction. Within the closed-form `_compute_K_per_point`, the `s`-direction is incorporated through:

- `drmncds`, `dzmnsds`, `dnumnsds` arrays (and the asym counterparts) supplied by the caller (the CPU wrapper's `BoozerRadialInterpolant.compute_K` at `boozermagneticfield.py:671-685`).
- `dRds`, `dZds`, `dnuds` computed as Fourier sums of those `s`-derivative coefficients (`boozer_radial_interp.py:154, 158, 195, 197`).

Both the cos-stack and sin-stack include the `s`-derivative terms in the expected positions (verified against the C++ destructure at `boozerradialinterpolant.cpp:41-46`). Sign conventions are correct. No CRITICAL or HIGH finding.

`θ`-derivatives (`dRdtheta`, `dZdtheta`, `dnudtheta`) and `ζ`-derivatives (`dRdzeta`, ...) are computed by multiplying-in the appropriate `xm[im]` / `−xn[im]` factor before the matmul; these are then summed against `sin_a` (for cos-symmetric Fourier coeffs whose ∂θ flips parity) or `cos_a` (for sin-symmetric coeffs). The matrix of signs lines up byte-for-byte with the C++ inner loop.

`B_mn(s)` derivative parity in the consumer wrapper is **out of scope** for item 32 but is exercised in `tests/field/test_boozermagneticfield_jax_item33.py::test_modB_derivs_bundle_matches_individual_methods`.

### (d) Axis-singularity parity

**Out of scope for the audited kernel** — but worth noting since the audit prompt explicitly mentions axis regularity (`|m|`-dependent `√s` powers).

The CPU wrapper handles axis regularity via the `mn_factor = s^{|m|/2}` profile (BOOZ_XFORM convention for VMEC half-grid coefficients). The pattern:

1. **Multiply in** at spline construction time: `bmnc_splines[im] = InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :] * bmnc[im, :])` (`boozermagneticfield.py:595`). This stores `coeff(s) · s^{|m|/2}` as the spline, which is **smooth at s=0** even for `m ≠ 0`.
2. **Divide out** at evaluation time: `bmnc_half[im, :] = bmnc_splines[im](s)/mn_factor` (`boozermagneticfield.py:674`).

This is **the standard VMEC axis-regularity trick**, not a Chebyshev expansion. The JAX wrapper reproduces it exactly in `_normalize` and `_radial_normalized` helpers (`boozermagneticfield_jax.py:396-408`):

```python
def _normalize(values, mn_factor):
    return values / mn_factor

def _radial_normalized(spline_vals, dspline_vals, mn_factor, d_mn_factor):
    return (dspline_vals - spline_vals * d_mn_factor / mn_factor) / mn_factor
```

The kernel audited here (`boozer_radial_interp.py`) **receives** already-divided coefficients from the consumer wrapper and is unaware of the axis-regularity trick. The kernel does **not** apply any `√s` factor.

> **Axis-singularity FINDING (consumer wrapper, item 33 scope, INFO-level here):** Both CPU and JAX evaluators divide by `mn_factor = s^{|m|/2}` at evaluation. At literally `s=0` and `m≠0` this division is `0/0`. The CPU wrapper relies on `InterpolatedUnivariateSpline` smoothness and assumes callers never query `s=0` exactly. The JAX wrapper inherits the same assumption (the `ppoly_eval` JIT will produce `NaN/0` if `s=0` exactly). This is **not a JAX parity bug** — both sides have the same `s>0` precondition — but a defensive `s = jnp.where(s == 0, epsilon, s)` clamp in the JAX evaluator might be worth a separate audit item. Filed here for visibility only.

### (e) Consistency-identity parity

**Out of scope for the audited kernel.** The identities

- `B·∇ζ = G(s) / J` where `J = (G + ι·I)/B²` is the Jacobian
- `B·∇θ = I(s) / J`
- `J·B² = G(s) + ι(s)·I(s)`

are properties of the **Boozer coordinate construction**, not of these Fourier kernels. They are pre-computed by BOOZ_XFORM upstream and consumed as the `G`, `I`, `iota` profiles. The audited kernel uses `sqrtg = (G + ι·I)/B²` as the definition (`boozer_radial_interp.py:218`, matching C++ `:62`, `:129`). No consistency check is performed on the input profiles themselves.

If you want identity-level coverage, that lives in the upstream BOOZ_XFORM validation, NOT in this kernel set. **No JAX/C++ parity gap here.**

## Test Coverage Gaps

| Gap | Severity | Location |
| --- | --- | --- |
| Forward `fourier_transform_{odd,even}` lacks a closed-form (oracle-free) parity test. Currently the only check is against the C++ binding, which has the OpenMP data race documented in Finding 5. | LOW (defensive) | `test_boozer_radial_interp_jax_item32.py:307-336` |
| `_compute_K_per_point` is tested only **transitively** via `compute_kmns`/`compute_kmnc_kmns`. No direct same-state oracle parity test against the C++ inner-loop computation of `K`. The current direct tests check jaxpr `dot_general` counts (`:171-200`) but not the K values. | LOW | `test_boozer_radial_interp_jax_item32.py` |
| No edge-case test for `num_modes == 1` (only the DC mode). C++ would do nothing for `compute_kmns` (the inner loop is `for im=1; im<1`), but the JAX path still constructs `(num_points, 1)` matrices and matmuls. Worth a sanity check that no out-of-bounds index is produced. | INFO | n/a |
| No edge-case test for `num_points == 0` or `num_surf == 0` (empty arrays). | INFO | n/a |
| No mixed-precision test. The kernels are implicitly float64 (NumPy/scipy defaults), but JAX's `jnp.where`/`einsum` will silently promote/demote if a caller passes float32. A `dtype` regression test would be defensive. | INFO | n/a |
| No bit-identity test under `XLA_FLAGS=--xla_gpu_deterministic_ops=true` or equivalent. This is the published parity-lane policy for CUDA (CLAUDE.md "GPU reproducibility policy fields"). The current tests run on CPU JAX only. | INFO (consistent with parity-ladder lanes) | runtime |
| No test asserting that the JAX kernels are `jax.jit`-traceable WITHOUT a host roundtrip when fed `jnp` arrays (the existing `test_kernels_are_jit_compatible` accepts NumPy arrays, so it triggers a host→device transfer). | INFO | `:561-600` |

## Recommended Actions (ordered by severity)

### MEDIUM

1. **Add a closed-form oracle test for `fourier_transform_odd` / `fourier_transform_even`.** Construct `K[ip] = sin(angle[ip, m*])` for a fixed `m* ∈ [1, num_modes)`, run `fourier_transform_odd`, assert the result is the unit basis vector `e_{m*}` to `direct_kernel` tolerance. Symmetric test for the cos branch with `K[ip] = cos(angle[ip, m*])` including `m*=0`. Pattern already exists for the **inverse** transform (`test_inverse_fourier_transform_reconstructs_pure_modes`, `:442-461`); replicate it for the forward transform. Rationale: insulates the JAX parity gate from the OpenMP data race in C++ `fourier_transform_*` (Finding 5).

2. **File a C++ upstream bug** for the `kmns(im) +=` / `norm +=` data race in `fourier_transform_{odd,even}`. The fix is `#pragma omp parallel for reduction(+:norm)` plus a per-mode-local accumulator for `kmns(im)`. Not blocking for the JAX port (JAX is correct), but the in-tree simsoptpp build is the C++ oracle of record and a Linux Linux CI run with `OMP_NUM_THREADS=8` could non-deterministically break this parity test.

### LOW

3. **Optional: add a direct `_compute_K_per_point` parity test against a hand-rolled NumPy reference** (one isurf, one ip, the same closed-form K formula computed in pure NumPy double-loop). This would catch sign/typo regressions in the per-point K that today are only detected via the surface-projection parity tests.

4. **Optional: rename `inverse_fourier_transform_{odd,even}` dispatch wrappers** to make it explicit that they are NOT JIT-decorated. A short docstring line "Not JIT-decorated — dispatches at Python time on `kmns.ndim`. The 1D and 2D variants are JIT-decorated." would help maintainers who try to call them under `vmap`.

### INFO

5. **(Out of scope for item 32) Audit the axis-singularity behaviour at `s = 0`** in `boozermagneticfield_jax.py::_eval_modB` (and siblings) by injecting a clamp `s = jnp.where(s == 0, eps, s)` or by switching to a smooth basis (e.g. transform `s ← √s` in the spline construction). The current behaviour is "don't query s=0", same as upstream CPU.

6. **(Out of scope for item 32) Add an identity-level integration test** in `test_boozermagneticfield_jax_item33.py` that compares `wrapper.modB() * sqrtg` against `G + ι·I` at a few sample points. Closes the audit-prompt requirement for `B·∇ζ = G` style identity coverage without depending on the kernel-level interp.

7. **(Cosmetic)** Replace positional `cos_values[:, 0]` indexing in the stellsym `_compute_K_per_point` with a `NamedTuple` or dataclass keyed by quantity name (`B`, `R`, `dRdtheta`, ...). Reduces the maintenance hazard called out in Finding 10.

## Conclusion

The six JAX Fourier helper kernels in `simsopt/jax_core/boozer_radial_interp.py` are **a correct, performant, GPU-purity-compliant port** of `src/simsoptpp/boozerradialinterpolant.cpp`. The math, the DC convention, the `1/(2π²)` / `1/(4π²)` normalisation factors, and the 1D-vs-2D `kmns` polymorphism all reproduce the C++ binding byte-for-byte at `direct_kernel` tolerance (`rtol=1e-10`, `atol=1e-12`). No CRITICAL or HIGH findings on the JAX side.

The single MEDIUM finding is a **C++ OpenMP data race in `fourier_transform_{odd,even}`** that JAX does NOT inherit; the recommended hardening is a closed-form oracle test for the forward transform that does not depend on the C++ side.

The audit-prompt items related to **radial basis (Chebyshev / spline in √s), axis singularity, and `B·∇ζ = G` identities are correctly out of scope** for the audited file — they live in the consumer wrapper (item 33, `BoozerRadialInterpolantJAX` in `simsopt/field/boozermagneticfield_jax.py`) and are covered by `tests/field/test_boozermagneticfield_jax_item33.py`.
