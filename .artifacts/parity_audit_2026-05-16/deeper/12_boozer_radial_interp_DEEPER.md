# PRIORITY 12 DEEPER PARITY AUDIT — `boozer_radial_interp.py` (item 32) and `BoozerRadialInterpolantJAX` (item 33)

Date: 2026-05-16
Auditor: second-pass deeper audit
Target: `src/simsopt/jax_core/boozer_radial_interp.py` (559 lines) and the radial reconstruction
consumer `src/simsopt/field/boozermagneticfield_jax.py` (1581 lines, item 33 wrapper) versus
`src/simsoptpp/boozerradialinterpolant.cpp` and `src/simsopt/field/boozermagneticfield.py`.

First-pass scope reminder: the six Fourier helper kernels in item 32 are parity-clean at
`direct_kernel` tolerance and the OMP race in `fourier_transform_{odd,even}` was already
flagged MEDIUM. The deeper pass below catalogues the issues a forward-formula audit could
have missed, with emphasis on the radial-reconstruction wrapper (item 33) and on
identity-level gaps the first pass punted to that wrapper.

---

## Summary table

| Severity | Finding | Where |
|----------|---------|-------|
| HIGH | `B·∇ζ = G(s)`-class Boozer identities are never tested in item 32 OR item 33; first-pass cross-reference is a dead link. | tests/field/test_boozermagneticfield_jax_item33.py whole file |
| HIGH | The `rescale=True` `mn_factor` is `s^{-m/2}` (negative exponent), not `s^{|m|/2}`. The task brief and item-32 docstring mislead about which side blows up. The JAX consumer's actual axis behavior depends entirely on the spline EXTRAPOLATION of `s^{-m/2}` outside `[s_half_mn[0], 1]` — and there is zero test of `s ∈ [0, s_half_mn[0])`. | boozermagneticfield.py:492-497, boozermagneticfield_jax.py:396-408 |
| HIGH | When `enforce_qs=True` filters a Fourier mode by writing zero into the K and B splines (CPU lines 591-593, 623-625, 710-713), the JAX wrapper inherits this through the frozen state, **but the test suite never constructs a `BoozerRadialInterpolant` with `N=...`** so the filtered-mode path is wholly untested in item 33. | boozermagneticfield_jax.py:303-341; tests/field/test_boozermagneticfield_jax_item33.py:81,89,98 |
| MEDIUM | `enforce_vacuum=True` overrides K, G, I, dGds, dIds (CPU lines 567-571, 365-367) but the JAX wrapper has no metadata flag for vacuum and inherits values through splines only. The test suite never exercises `enforce_vacuum=True`. | boozermagneticfield.py:365-371, 567-571 |
| MEDIUM (confirmed; carryover) | OMP data race on `kmns(im) +=` and `norm +=` in C++ `fourier_transform_{odd,even}` (cpp:147-156, 165-175). The race only fires for `OMP_NUM_THREADS > 1`, and these C++ symbols are NOT invoked from the production radial-reconstruction path. JAX is race-free by construction. | boozerradialinterpolant.cpp:147-156, 165-175 |
| MEDIUM | Item-33 wrapper has a stale-state risk: the upstream `BoozerRadialInterpolant` has no setters, but `psi0`, `enforce_qs`, `enforce_vacuum`, `N`, `no_K` are mutable attributes. The JAX wrapper freezes them at `__init__`; any post-construction mutation on the upstream does not propagate. Module docstring claims this, but no regression test asserts it. | boozermagneticfield_jax.py:9-19, 832-836 |
| MEDIUM | No test compares the JAX wrapper to the CPU wrapper across the documented `s ∈ [0, s_half_mn[0])` extrapolation regime; the existing fixture starts at `s = 0.05` while typical `s_half_mn[0]` (no `ns_delete`) is `1/(2*ns_in) ≈ 0.01` (well above 0.05 only sometimes). | tests/field/test_boozermagneticfield_jax_item33.py:70-74 |
| LOW | Inverse-FT 1D vs 2D polymorphism dispatch uses Python `if kmns.ndim` — fine for eager calls, but breaks `jax.vmap` over the leading "polymorphism" axis. Not currently used as a polymorphism point inside JIT — but the wrapper imports the polymorphic dispatcher (`boozermagneticfield_jax.py:57-60`) rather than the explicit `*_2d` variant, gaining no compile-time safety. | boozer_radial_interp.py:471-490, 528-545 |
| LOW | `fourier_transform_odd/even` JAX implementation guards `denom=0` for `im=0` (mask + safe_denom) but does NOT guard `denom=0` for `im>=1`. The C++ also does NOT guard. Physically, `denom = sum_ip sin(angle)^2 = 0` if every sample lies on a zero of the sine — possible only for degenerate (theta, zeta) grids. Latent NaN. | boozer_radial_interp.py:390-396, 400-417; boozerradialinterpolant.cpp:154,173 |
| LOW | Non-stellsym degenerate case (asym coefficient arrays all-zero) is never tested in item 32; the JAX kernel's `have_asym = rmns is not None` branch (line 111) selects the asym path unconditionally when the caller passes the array, even if every entry is zero, so the kernel does extra work but the result is correct. Latent dead computation. | boozer_radial_interp.py:111-148 |
| LOW | `compute_kmnc_kmns` and `compute_kmns` shadow each other's coefficient ordering; the asym branch (lines 113-162) uses a 12-row stack and the stellsym branch (lines 164-199) uses a different 7+5 split. Both correct, but the maintainability risk is real. | boozer_radial_interp.py:111-200 |
| LOW | `inverse_fourier_transform_{odd,even}` `ValueError` message is informative ("ndim 1 or 2, got ndim=3; shape=(...)"). Confirmed test coverage at tests/jax_core/test_boozer_radial_interp_jax_item32.py:486-494. | boozer_radial_interp.py:488-490, 543-545 |
| INFO | The strict K post-scaling `K = K_raw * dθ * dζ * nfp / psi0` (CPU lines 699, 704) is performed in Python BEFORE building `kmns_splines`. The JAX wrapper reads the already-scaled splines — there is no second JAX-side multiplication. So `psi0` agreement is a "passive" issue: if the user changes `bri.psi0` after construction, the JAX `_psi0` field and the K splines diverge. | boozermagneticfield.py:699-717; boozermagneticfield_jax.py:833 |

---

## 1. The B·∇ζ = G(s) Identity is Tested NOWHERE (HIGH)

The first-pass deferred the identity test "to item 33 wrapper test." I read
`tests/field/test_boozermagneticfield_jax_item33.py` end-to-end (239 lines). The full set of
named identity checks across the test module:

- `_compare_all_methods` (line 103) — JAX-vs-CPU parity on the public API surface.
- `test_modB_derivs_bundle_matches_individual_methods` (line 184) — internal bundle vs
  per-method consistency, NOT a Boozer-coordinate identity.

There is **no** assertion of `B·∇ζ = G(s)`, no `B·∇θ = I(s) + iota·G(s)` style check, no
`sqrt(g) = (G + iota·I)/B²` check, no `B = G ∇ζ + I ∇θ + K ∇ψ` reconstruction. The
geometry quantities (`R`, `Z`, `nu`, `dRdtheta`, ...) are all checked against CPU as
re-evaluations of the same closed-form formula — that is a "JAX agrees with CPU" check, NOT
a "the implementation respects Boozer coordinates" check.

Why this matters: the only thing that could reveal a sign-flip, mode-index-bug, or
half-grid/full-grid mismatch internal to BOTH backends is a CLOSED-FORM physics identity
that does not factor through either backend. The current item-33 test suite cannot
distinguish "JAX correctly reproduces a CPU bug" from "JAX is correct."

**Recommended gap-closure**: add a test fixture that, for an analytic Boozer field
(e.g., `BoozerAnalytic` or a hand-constructed `BoozerRadialInterpolant` with one mode),
explicitly checks `B · ∇ζ ≈ G(s)` at several `(s, θ, ζ)` points. The vector
identity in Boozer coordinates is

  B^ζ = (G + ι·I)/(B²) · g^{ζζ}  → simplifies to  B·∇ζ = G(s)/sqrt(g)·(...)

Practical form (from `boozermagneticfield.py:29-39`):

```
B(s,θ,ζ) = G(s) ∇ζ + I(s) ∇θ + K(s,θ,ζ) ∇ψ
sqrt(g)(s,θ,ζ) = (G + ι·I)/B²
```

Implies the contravariant component `B^θ = ι·B²/(G + ι·I)`, `B^ζ = B²/(G + ι·I)` —
both functions of `s` only. The audit verdict is that this is the right identity to
check, and absence of any such test is a coverage gap whose severity grows with how
much downstream tracing or guiding-centre work depends on the JAX wrapper.

---

## 2. `mn_factor` Convention: Negative Exponent (HIGH)

Reading `boozermagneticfield.py:490-497`:

```python
mn_factor = np.ones_like(bmnc)
d_mn_factor = np.zeros_like(bmnc)
mn_factor[self.xm_b == 1, :] = s_half_mn[None, :]**(-0.5)
d_mn_factor[self.xm_b == 1, :] = -0.5*s_half_mn[None, :]**(-1.5)
mn_factor[(self.xm_b % 2 == 1)*(self.xm_b > 1), :] = s_half_mn[None, :]**(-1.5)
d_mn_factor[(self.xm_b % 2 == 1)*(self.xm_b > 1), :] = -1.5*s_half_mn[None, :]**(-2.5)
mn_factor[(self.xm_b % 2 == 0)*(self.xm_b > 1), :] = s_half_mn[None, :]**(-1.)
d_mn_factor[(self.xm_b % 2 == 0)*(self.xm_b > 1), :] = -s_half_mn[None, :]**(-2.)
```

Exponents are negative. The brief and the item-32 docstring "mn_factor = s^{|m|/2} for
axis regularity" is upside-down. The actual mapping is:

- physical Fourier coefficient `f_{mn}(s) ~ s^{m/2}` near axis (regularity condition).
- `mn_factor(s) = s^{-m/2}` (capped at 1 for `m=0` and at `s^{-1}` for even `m ≥ 2`).
- spline-baked value: `spline_value = mn_factor * f_{mn}(s) ~ s^{-m/2} · s^{m/2} = O(1)`.
- evaluation: `f_{mn}(s_eval) = spline(s_eval) / mn_factor(s_eval)`.

So the strategy is to interpolate the well-behaved quantity `s^{-m/2} · f_{mn}(s)` instead
of the s-singular `f_{mn}(s)`. The spline of `s^{-m/2}` itself (`mn_factor_splines`) is
built from a sample of size `s_half_mn` which starts at the first half-grid point. For
typical VMEC `wout_n3are` the first half-grid point is roughly `s_in[0] = 1/(2*ns_in) ≈
0.004` (depending on `ns_in`).

**The non-obvious risk**: when the user passes `s_eval < s_half_mn[0]` (e.g., `s_eval =
0`), both the `spline` and `mn_factor` are polynomial-EXTRAPOLATED. The extrapolated
`mn_factor` is *finite*, not infinite — the cubic polynomial fit on the sampled tail of
`s^{-1.5}` continues with whatever curvature the cubic happens to have. The
extrapolated `spline_value` is similarly finite. Their ratio at `s = 0` is a finite
number but **NOT** the physical limit `lim_{s→0} f_{mn}(s) = 0`.

This means: **the JAX wrapper and the CPU wrapper agree byte-by-byte at `s = 0` (both
extrapolate the same scipy PPoly), but neither is physically correct.** The forward-parity
audit cannot detect this — it is by construction a JAX-vs-CPU comparison. Only a physics
identity (item 1) or a high-resolution `s → 0` limit check could reveal it.

**Test gap**: no test in item 33 exercises `s < s_half_mn[0]`. The fixture
`_make_evaluation_points` at `tests/field/test_boozermagneticfield_jax_item33.py:70-74`
starts at `s = 0.05`. For `wout_n3are_R7.75B5.7_lowres.nc` (small VMEC grid) the first
half-grid point is typically well above 0.001, so `s = 0.05` is solidly inside the spline
support. There is **no test of `s ∈ [0, s_half_mn[0])`** and consequently no parity
evidence for the axis-singularity-via-extrapolation regime.

Verdict: the JAX wrapper is parity-correct in the regime where it is tested. The
extrapolation regime is not tested, but the wrapper inherits whatever scipy PPoly returns,
so JAX-CPU parity in that regime should still hold; only the *physical correctness* is
uncertain and is inherited from the CPU side.

---

## 3. The `enforce_qs` Filtered-Mode Path is Untested (HIGH)

`BoozerRadialInterpolant.__init__(..., N=None, ...)` (line 335) and lines 370-372:

```python
if (N is not None):
    self.N = N
    self.enforce_qs = True
```

Then at spline construction lines 591-593, 623-625, 710-713:

```python
if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
    self.bmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, 0*bmnc[im, :], k=self.order))
```

So non-QS modes are zeroed-out at the spline level. The JAX wrapper's `freeze_boozer_radial_state`
at `boozermagneticfield_jax.py:303-341` blindly reads `upstream.bmnc_splines`, so this
filtering propagates correctly via the spline data.

However, the test suite at `tests/field/test_boozermagneticfield_jax_item33.py:77-100`
constructs three fixtures (`stellsym_bri_and_jax`, `stellsym_no_K_bri_and_jax`,
`asym_bri_and_jax`) — none pass `N=...`. So the QS-filter code path is exercised in
neither item 32 nor item 33. If a downstream user activates QS filtering on a
BoozerRadialInterpolant and constructs a `BoozerRadialInterpolantJAX`, there is zero
parity evidence.

The filtering is also applied to `kmns_splines` lines 710-713 (zero out non-QS K modes).
This interacts with the K-coefficient post-scaling (`* dθ * dζ * nfp / psi0`) — the
zero is preserved through the multiplication. The JAX wrapper inherits zero coefficients
correctly.

**Recommended**: add a parametrized variant of `test_stellsym_public_api_matches_cpu` that
sets `N=3` (matches typical helicity for low-res VMEC files) and confirms the JAX wrapper
zeroes the same non-QS modes.

---

## 4. The `enforce_vacuum=True` Path is Untested (MEDIUM)

`BoozerRadialInterpolant.__init__` at line 365-367:

```python
if (self.enforce_vacuum):
    self.no_K = True
```

And at lines 568-571:

```python
self.G_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.mean(G)*np.ones_like(self.s_half_ext), k=self.order)
self.I_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.zeros_like(self.s_half_ext), k=self.order)
self.dGds_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)
self.dIds_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)
```

So vacuum means `G = mean(G)` (constant), `I = 0`, `dGds = 0`, `dIds = 0`, `K = 0`. The
JAX wrapper has a `no_K` meta-flag (boozermagneticfield_jax.py:155, 191) but **no
analogous `enforce_vacuum` flag**. The vacuum constants come through the splines
naturally.

The JAX wrapper exposes `no_K` as a public property (line 871-873). Test
`test_stellsym_no_K_public_api_matches_cpu` exercises `no_K=True` directly (line 86-91),
but the `enforce_vacuum=True` route (which also implies `no_K=True` and zeroes G/I
behavior) is never tested. The vacuum case is the common starting point for many
stellarator optimizations (Stage 2), so a missing-test gap here is non-trivial.

---

## 5. OMP Data Race in `fourier_transform_{odd,even}` (MEDIUM, carryover)

C++ at `boozerradialinterpolant.cpp:147-156` and `165-175`:

```cpp
double norm;
for (int im=1; im < num_modes; ++im) {
  norm = 0.;
  #pragma omp parallel for
  for (int ip=0; ip < num_points; ++ip) {
    kmns(im) += K(ip)*sin(...);    // RACE: kmns(im) is shared across threads
    norm += pow(sin(...), 2.);     // RACE: norm is shared across threads
  }
  kmns(im) = kmns(im)/norm;
}
```

Two race conditions: `kmns(im) +=` and `norm +=`. Neither has a `reduction(+:kmns,norm)`
clause. Standard race semantics: results are non-deterministic when `OMP_NUM_THREADS > 1`.

**Triggerability**: I did not run a live `OMP_NUM_THREADS=8` repro in this audit (would
require an interactive simsoptpp environment), but the race fires by construction when:

1. `num_points` is large enough that OpenMP actually parallelises the inner loop (typically
   requires `num_points >= OMP_NUM_THREADS * grain`, default grain ~32).
2. `OMP_NUM_THREADS > 1`.

**Why this is dormant in production**: `fourier_transform_odd/even` are imported from
`simsoptpp` only by `tests/jax_core/test_boozer_radial_interp_jax_item32.py` — they are
not invoked anywhere in `boozermagneticfield.py` (the production radial-reconstruction
path uses `compute_kmns` / `compute_kmnc_kmns` / `inverse_fourier_transform_{odd,even}` —
NOT the `fourier_transform_*` forward projectors). Grep confirms:

```
src/simsopt/field/boozermagneticfield.py:692:  kmnc_kmns = sopp.compute_kmnc_kmns(...)
src/simsopt/field/boozermagneticfield.py:701:  kmns = sopp.compute_kmns(...)
src/simsopt/field/boozermagneticfield.py:730:  sopp.inverse_fourier_transform_odd(K[:, 0], kmns, ...)
src/simsopt/field/boozermagneticfield.py:748:  sopp.inverse_fourier_transform_even(...)
```

The other C++ kernels (`compute_kmns`, `compute_kmnc_kmns`, `inverse_fourier_transform_*`)
are race-free: `compute_kmns` parallelises the OUTER surface loop (cpp:88) with each
thread owning its `isurf` column, and `inverse_fourier_transform_*` parallelise the
INNER point loop on `K(ip)` which is a per-iteration target.

**Recommended fix**: add `reduction(+:kmns,norm)` to the OMP pragma in
`fourier_transform_{odd,even}`. JAX inherits no defect; this is purely C++ housekeeping.

---

## 6. Stale State on Upstream Mutation (MEDIUM)

The wrapper module docstring claims "Frozen state semantics: mutating the wrapped CPU
instance after construction does not propagate to the JAX wrapper" (lines 17-19). This is
*true by construction* because `freeze_boozer_radial_state` reads `pp.c` and `pp.x` once
and copies into `jnp.asarray` (lines 274-277), so the JAX state is decoupled.

However, two latent risks:

1. **psi0 mutation**: `bri.psi0` is a plain attribute (no setter on
   `BoozerRadialInterpolant`, only on `BoozerAnalytic` at line 199). If the user reaches
   in and does `bri.psi0 = X`, the JAX wrapper's `_psi0 = 2.0` (frozen at line 833)
   stays at the OLD value, and the K splines were ALSO baked with the old psi0 at line
   704 (`kmns = kmns*...*nfp/self.psi0`). So both the JAX cache and the K spline data
   would be self-consistent (both reflect the old psi0), but the upstream `bri.psip()`
   would diverge from `wrapper.psip()`. No test catches this.

2. **xm_b / xn_b mutation**: `bri.xm_b` could in principle be re-assigned. The JAX wrapper
   freezes `xm`/`xn` into a JAX array (line 300-301). If `bri.xm_b` changes, the JAX
   wrapper's modes are stale. No test catches this. (Practically, this requires the user
   to re-run `bri.booz.register(...)` and `bri.init_splines()` — equivalent to
   reconstructing the CPU instance, at which point reconstructing the JAX wrapper is the
   obvious right move.)

**Recommended**: add a regression test that mutates `bri.psi0` and confirms the JAX
wrapper still returns the old value (positive evidence of frozen-state isolation).

---

## 7. The "Half-Grid vs Full-Grid" Spline Decision (INFO)

The first-pass mentioned "half-grid vs full-grid spline construction." Reading the CPU
code:

- Half-grid splines (`InterpolatedUnivariateSpline(self.s_half_ext, ...)` at lines
  563-572, 586-590, 595, etc.): for `G`, `I`, `iota`, `bmnc`, `rmnc`, `zmns`, `numns`,
  `bmns`, `rmns`, `zmnc`, `numnc`, `mn_factor`, `d_mn_factor`.
- Full-grid splines (`InterpolatedUnivariateSpline(s_full[1:-1], ...)` at lines 565-566,
  573, 593, 599, 606-608, 625, 631, 638-640): for `dGds`, `dIds`, `diotads`, `dbmncds`
  (non-rescale only), `dbmnsds` (non-rescale only), `dnumnsds`, `drmncds`, `dzmnsds`, and
  the asym counterparts.

For `rescale=True`, the derivative splines are built via `<spline>.derivative()` (lines
597, 602-604, 629, 634-636) which uses the SAME breakpoints as the parent, so half-grid
breakpoints in the rescale path. For `rescale=False`, derivative splines use full-grid
breakpoints.

The JAX wrapper does NOT distinguish — it reads whatever PPoly the CPU built. So the
half-grid vs full-grid behavior of the JAX wrapper is byte-identical to the CPU. No
JAX-only spline implementation is involved (scipy's `InterpolatedUnivariateSpline`
remains the source of truth, and `boozermagneticfield_jax.py:247-249` converts it via
`PPoly.from_spline` at construction time, so the radial interpolation in the JAX path
runs on host scipy data baked into JAX arrays — the actual evaluation kernel
`ppoly_eval` in `jax_core/boozer_fixed_state.py:192-206` is a pure JAX Horner
evaluation).

This means the JAX path is partially on-device (the polynomial evaluation is JAX) and
partially off-device (the polynomial construction is scipy). Acceptable: this is the
"freeze CPU splines, evaluate on device" design pattern documented in the module
docstring.

**Discontinuity at half-grid boundaries**: `InterpolatedUnivariateSpline` of order `k` is
C^{k-1} continuous everywhere within its support; `PPoly.from_spline` preserves this
exactly. The JAX `ppoly_eval` does naive Horner per-segment, so the JAX continuity is
also C^{k-1}. No risk.

---

## 8. K-Coefficient Post-Scaling Convention (INFO, confirmed clean)

CPU lines 699 and 704:

```python
kmnc = kmnc*dtheta*dzeta*self.booz.bx.nfp/self.psi0
kmns = kmns*dtheta*dzeta*self.booz.bx.nfp/self.psi0
```

This rescales the raw Fourier-projection output from `simsoptpp.compute_kmnc_kmns` /
`compute_kmns` (which only inserts `1/(2π²)` or `1/(4π²)` normalisation, NOT the
quadrature step or the `1/ψ_0` factor). The final K coefficients are stored in
`kmnc_splines` / `kmns_splines` (lines 715, 717) multiplied by `mn_factor`.

The JAX kernel `compute_kmns` in `boozer_radial_interp.py:223-278` faithfully reproduces
the inner `1/(2π²)` normalisation (line 274) and emits the raw kmns without applying
`dθ·dζ·nfp/ψ_0`. The CPU consumer at line 704 applies this scaling. The JAX consumer at
line 337 of `boozermagneticfield_jax.py` does NOT re-scale — it reads the already-scaled
spline payload from the CPU instance. So `ψ_0` does NOT appear in the JAX evaluation
path at all (only in `_eval_psip` indirectly, via the precomputed `psip` spline).

**Consequence**: any change to `bri.psi0` after construction silently corrupts the JAX
wrapper's K evaluation, as flagged in (6).

---

## 9. DC-Row Zeroing Convention (confirmed consistent, INFO)

Verification across all six kernels:

| Kernel | DC (im=0) behavior |
|--------|---------------------|
| `compute_kmns` | sin contribution at im=0 zeroed (boozer_radial_interp.py:273; mirrors cpp:132 `for (im=1; ...)`) |
| `compute_kmnc_kmns` cos | im=0 INCLUDED with `1/(4π²)` scaling (lines 349-353; mirrors cpp:71) |
| `compute_kmnc_kmns` sin | im=0 zeroed (line 357; mirrors cpp:67 `if (im > 0)`) |
| `fourier_transform_odd` | im=0 zeroed (line 394-396; mirrors cpp:147 `for (im=1; ...)`) |
| `fourier_transform_even` | im=0 INCLUDED (no zeroing; mirrors cpp:166 `for (im=0; ...)`) |
| `inverse_fourier_transform_odd` 1D/2D | im=0 zeroed (lines 442, 466; mirrors cpp:184, 191) |
| `inverse_fourier_transform_even` 1D/2D | im=0 INCLUDED (lines 506-507, 524-525; mirrors cpp:206, 213) |

All six match. Tests at item-32 lines 244 and 299 directly assert the DC row is exactly
zero for the sin variants — confirmed clean.

---

## 10. C++ UB Beyond the Known OMP Race (INFO, no new findings)

End-to-end re-read of `boozerradialinterpolant.cpp` (221 lines):

- **Uninitialized scalars**: every scalar in the inner loop (B, R, dRdtheta, ..., dnudzeta)
  is initialized to `0.` at the start of each `ip` iteration (lines 23-34, 91-102).
  Clean.
- **Signed/unsigned mixing**: `int num_modes`, `int num_surf`, `int num_points`,
  `int isurf`, `int ip`, `int im`, `int dim` — all signed. xtensor `shape(...)` returns
  `size_t` (unsigned); assigning to `int` is implementation-defined for large arrays but
  fine in practice. Clean.
- **Division-by-zero in compute_kmn?**: line 62-63 and 129-130: `sqrtg = (G(isurf) +
  iota(isurf)*I(isurf))/(B*B)`. If `B == 0` at some quadrature point, this is `+inf`.
  Then `K = (gszeta + iota*gstheta)/sqrtg = 0 * inf` if the numerator also vanishes
  (unlikely) — or `0` if numerator is finite. The JAX kernel `_compute_K_per_point` at
  line 218 has the IDENTICAL formula and the same potential for `B=0` to produce
  `nan`/`inf`. Physically, `B(s,θ,ζ) > 0` everywhere on a flux surface for any sane
  field, and `bmnc[0,:] ~ 1.0` in the test fixtures keeps B safely positive.
  Latent edge case, not a real bug.
- **Norm division-by-zero in fourier_transform_odd**: line 154 `kmns(im) = kmns(im)/norm`
  with no `norm > 0` guard. If every sample `sin(angle)` is zero (e.g., a mode is
  identically zero on the chosen grid), `norm = 0` and the result is `nan`. The JAX
  kernel guards `im=0` only (line 394-396) and inherits the same `nan` for `im>=1`. The
  test fixtures use random `(thetas, zetas)` so the risk is zero in practice. (See item
  11 in the brief.)

---

## 11. `norm = 0` in `fourier_transform_odd` (LOW)

Brief item 11 asks when `norm = 0` is physically achievable. The denominator is

```
norm = sum_{ip=0}^{N-1} sin(xm[im]·θ_ip - xn[im]·ζ_ip)²
```

For random `(θ_ip, ζ_ip)` this is bounded below by `N/2` in expectation (mean of `sin²` is
0.5 on `[0, 2π]`). The pathological case is:

- `xm[im] == xn[im] == 0` (the DC mode) → every angle is zero, every sin is zero,
  norm = 0. But im=0 is skipped, so unaffected.
- All `(θ_ip, ζ_ip)` lie on the zero-set of one mode — e.g., uniform `θ_ip = π·k/xm[im]`
  for integer k. Possible on a deliberately adversarial structured grid; not realistic for
  Boozer-coordinate quadrature.

Neither path is exercised by the test suite; the JAX kernel's behaviour matches the C++
(both return `nan` for affected entries). No production-path consumer of
`fourier_transform_odd` exists, so the latent NaN risk is purely diagnostic.

---

## 12. 1D vs 2D vs ndim ≥ 3 Polymorphism (LOW)

`inverse_fourier_transform_odd/even` in `boozer_radial_interp.py:471-490` and `528-545`:

```python
def inverse_fourier_transform_odd(kmns, xm, xn, thetas, zetas):
    if kmns.ndim == 1:
        return inverse_fourier_transform_odd_1d(kmns, xm, xn, thetas, zetas)
    if kmns.ndim == 2:
        return inverse_fourier_transform_odd_2d(kmns, xm, xn, thetas, zetas)
    raise ValueError(
        f"kmns must have ndim 1 or 2, got ndim={kmns.ndim}; shape={tuple(kmns.shape)!r}"
    )
```

This is a Python `if` outside the JIT. Calling with `ndim=3` raises a clear `ValueError`
with the offending shape included. Confirmed by test at item-32 line 486-494.

**Subtle issue**: the `if` runs at Python tracing time. If used inside `jax.vmap` over a
"polymorphism axis", vmap would see the result as a single concrete `ndim` and pick the
wrong branch silently. Not currently a hazard because the wrapper at
`boozermagneticfield_jax.py:419-424` always calls with a 2D coefficient table (the column
gather `_column_at(s, ...)` returns shape `(num_modes, num_points)`), so the 1D branch is
dead code inside the wrapper. Tests at item-32 lines 346-364 exercise the 1D branch
directly.

---

## 13. Stellsym vs Non-Stellsym Split (LOW, confirmed correct)

The `_compute_K_per_point` helper at line 78-219 splits on `have_asym = rmns is not
None`. For a degenerate non-stellsym surface (where every asym coefficient happens to be
zero), the caller could:

- Pass `rmns=zeros_like(...)`, etc. → `have_asym=True` → the JAX kernel runs the 12-row
  asym path with all-zero contributions. Result is correct; cost is ~2x the stellsym
  path.
- Construct `BoozerRadialInterpolant` with `bri.stellsym=False` (manual override). The
  CPU spline builder reads `self.booz.bx.bmns_b` etc.; if those are zero arrays, the
  CPU and JAX paths both populate zero splines. The JAX kernel's `not state.stellsym`
  branch runs and adds zero contributions. Correct.

No degenerate-asym tests exist, but the parametric coverage on
`test_compute_kmnc_kmns_matches_cpp` (item 32 line 252-298) and
`test_asym_public_api_matches_cpu` (item 33 line 212-216) provides indirect coverage by
random asym coefficients of magnitude `~0.1`. A test with `rmns = zeros` and
`bri.stellsym = False` would be a useful "no info loss in asym dispatch" sanity check.

---

## 14. Stale-State Risk Around `kmnc_splines` for Stellsym (LOW)

`freeze_boozer_radial_state` at lines 333-341:

```python
if no_K:
    kmns = _zeros_like_profile(bmnc)
    kmnc = _zeros_like_profile(bmnc)
else:
    kmns = _mode_profile_stack(upstream.kmns_splines)
    if stellsym:
        kmnc = _zeros_like_profile(kmns)
    else:
        kmnc = _mode_profile_stack(upstream.kmnc_splines)
```

For stellsym, the JAX wrapper *fabricates* a zero `kmnc` profile (`_zeros_like_profile`)
because the CPU class doesn't even define `upstream.kmnc_splines` in the stellsym case
(grep confirms no `self.kmnc_splines = []` outside `if not self.stellsym` at line 708).
This is correct because `_eval_K` and `_eval_dKd{theta,zeta}` gate the kmnc contribution
behind `if not state.stellsym` (lines 705-759). Belt-and-suspenders defensive design;
clean.

---

## 15. Untested Edge-Case Inventory

For a complete picture, here is the inventory of edge cases that the current item-32 /
item-33 test suites do not exercise. None are necessarily bugs; they are coverage gaps a
forward-formula audit cannot evaluate.

1. **`s = 0` and `s` near `s_half_mn[0]`** (axis-extrapolation regime).
2. **`s > 1`** (boundary-extrapolation regime).
3. **`enforce_qs=True` via `N=...`** (mode-filter zeroing on bmnc/bmns/kmns/kmnc splines).
4. **`enforce_vacuum=True`** (constant G, zero I, zero K, zero dGds/dIds).
5. **`ns_delete > 0`** (rescale path with axis-near points dropped).
6. **`rescale=False`** path (the existing fixtures all use `rescale=True`; the unrescaled
   path with `mn_factor = ones` is untested in item 33).
7. **`order=1` or `order=5`** spline orders (existing fixtures all use `order=3`).
8. **mpi-aware split** (`self.mpi is not None` branch — JAX wrapper doesn't read
   per-rank state, but the test should at least pass on rank ≠ 0).
9. **`bri.psi0` mutation after wrapper construction**.
10. **`bri.xm_b`/`xn_b` rebinding after wrapper construction**.
11. **`points` with `ndim ≠ 2` or `shape[1] ≠ 3`**: covered (line 165 negative test in
    `test_set_points_rejects_bad_shape`). Clean.
12. **`points` with `n=0` rows**: not tested. The JAX evaluators all accept zero-row
    inputs; the result is an empty array. Latent.
13. **`B·∇ζ = G(s)` and `B·∇θ` identity** (see item 1).
14. **`bri` constructed from a `Boozer` instance** (the `elif isinstance(equil, Boozer)`
    branch at boozermagneticfield.py:343-359) is untested in item 33.

---

## 16. Verification Sketch (no live runs performed)

The following micro-experiments would close the most critical gaps. Sized for a single
follow-up commit:

1. **B·∇ζ identity check**: construct `bri = BoozerAnalytic(etabar, B0, N, G0, psi0,
   iota0)` and call `wrapper.G()` vs a Boozer-coordinate evaluation of B in (R, φ, Z) and
   the contravariant relation. Tolerance: `rtol=1e-10` direct_kernel.
2. **Axis-extrapolation parity**: extend `_make_evaluation_points` to include `s ∈ {0.0,
   1e-15, s_half_mn[0]/2}` and confirm JAX-CPU parity across the extrapolation regime
   at `direct_kernel` tolerance.
3. **`enforce_qs=True` parity**: construct `bri = BoozerRadialInterpolant(vmec, ..., N=3)`
   and re-run `_compare_all_methods` to confirm the JAX wrapper agrees with the CPU
   class for filtered modes.
4. **psi0 mutation invariance**: `bri.psi0 = 999.0`; assert `wrapper.modB()` is unchanged
   (positive evidence of frozen-state isolation).
5. **OMP race fix**: add `reduction(+:kmns(im))` and `reduction(+:norm)` to
   `fourier_transform_odd/even` C++ pragmas. JAX is unaffected.

---

## 17. Verdict

The item-32 Fourier helper kernels are parity-clean. The first-pass forward-formula audit
correctly identified the OMP race (now confirmed dormant in production paths). The deeper
audit surfaces three HIGH-severity COVERAGE gaps (not bugs):

- The B·∇ζ identity check is missing across the entire `BoozerRadialInterpolant{JAX,}`
  parity stack.
- The axis-extrapolation regime (`s < s_half_mn[0]`) is untested.
- The `enforce_qs` filtered-mode path is untested.

Plus one MEDIUM-severity carryover (the OMP race) and several documentation/coverage
gaps. No new correctness bugs are evidenced in the JAX kernels; the JAX wrapper inherits
the same numerical behaviour as the CPU wrapper across the regimes tested. The risks I
have flagged are about regimes that are not tested.

The most actionable single change is the OMP race fix; the most consequential single
change is the addition of a B·∇ζ identity assertion to item 33's test module, which would
shore up the parity claim from "JAX agrees with CPU" to "JAX agrees with Boozer-coordinate
physics."
