# PRIORITY 13 — Analytic Field Kernels DEEPER Parity Audit

| Item | Value |
|------|-------|
| Audit date | 2026-05-16 |
| Branch | gpu-purity-stage2-20260405 |
| Audit author | Claude (Opus 4.7, 1M context) |
| Pass | SECOND-PASS (deeper) |
| Scope | Hunts not covered by first-pass forward-formula parity audit `13_analytic_fields.md` |
| Files in scope | `src/simsopt/jax_core/analytic_fields.py` (896 lines); `src/simsoptpp/dommaschk.cpp` (533); `src/simsoptpp/reiman.cpp` (107); both headers; `src/simsopt/field/magneticfieldclasses.py:752-844`; `src/simsopt/field/dommaschk_jax.py`; `src/simsopt/field/reiman_jax.py`; `tests/jax_core/test_analytic_fields_item11.py`; `tests/field/test_magneticfieldclasses_jax_item15.py`; `tests/field/test_magneticfields.py:642-1031` |

---

## Executive summary

Second-pass audit confirms no CRITICAL or HIGH severity findings beyond those already raised by the first pass. The forward-formula parity is solid. **Eight new findings** ranging from MEDIUM to INFO surfaced from drilling into recursion structure, branch-cut behaviour under autodiff, JIT-closure mutability, derivative-bound boundaries, and validation surfaces. Two of these (D-5 default constructor with `n=0`, D-1' the merged-monomials ULP issue revisited with worst-case bound) deserve action before the next release; the rest are low-impact reporting gaps.

### Severity roll-up

| ID | Severity | Title | Action |
|---|---|---|---|
| D-1' | MEDIUM | `_accumulate_terms` worst-case ULP bound, validated by hand | document and add a guarded fixture |
| D-4 | MEDIUM | `jax.grad(reiman_B)` w.r.t. `points` is NaN at the magnetic axis ring and at the φ-branch cut | document, add NaN-tolerant integration test |
| D-5 | LOW  | `DommaschkSpec(m=0, n=0)` (the default constructor) does NOT round-trip through the wrapper after a coeff mutation | document; mutability contract should be tightened |
| D-6 | LOW  | `lru_cache` on `_dommaschk_term_bundle` keyed only on `(m, n)`, never invalidated even if the user redefines `_DommaschkTerm` (informational) | document |
| D-7 | LOW  | No validation of `k_theta` in `ReimanSpec`: `kth ≤ 0` leads to silent garbage (Inf, NaN) under autodiff | add validation |
| R-3 | LOW  | `arctan2` branch-cut → `jax.grad(reiman_B)` returns NaN for points on the magnetic axis `(R, 0, 0)` | document, add tolerant test |
| T-1' | INFO | No `trace(dB) = 0` direct assertion on JAX Dommaschk, no `B_φ = -1` direct assertion on JAX Reiman, no odd-`k` parity, no `jax.grad`-over-coefficients parity | add 4 tests |
| T-2' | INFO | First-pass D-1 reduction-order divergence is currently buried in a comment; explicit `pytest.warns` / `pytest.skip` / tracked relaxed-tol marker would close the audit trail | tighten test scaffolding |

---

## 1 — Dommaschk recursion depth and `Vmn` / `Dmn` recursion vs direct formula

**Prompt item 1, 7 (recursion depth; recursion vs direct formula).**

### 1.1  Is there a recursion?

There is **no recursion** in either the JAX kernel or the C++ kernel. Both implementations evaluate `Dmn` and `Nmn` as **flat double sums** over `(k, j)`:

```c
// dommaschk.cpp:60-71 (Dmn)
for (k = 0; k <= floor(n/2); k++) {
    sumD = 0.0;
    for (j = 0; j < k + 1; j++) {
        sumD += <three monomial contributions, indexed by (k, j)>;
    }
    y += (pow(Z, n - 2*k) / tgamma(n - 2*k + 1)) * sumD;
}
```

```python
# analytic_fields.py:128-155 (_dmn_terms)
for k in range(n // 2 + 1):
    z_pow = n - 2 * k
    outer = 1.0 / math.gamma(z_pow + 1)
    for j in range(k + 1):
        <three _accumulate_terms calls>
```

The depth of nesting in both is at most 2 levels of bounded integer loops. **No memoization** of `Dmn(m, n')` for `n' < n` exists; each `(m, n)` is independently expanded. **There is no partition between "direct formula" and "recursion" branches**: the same double-sum is used for all `(m, n)`.

### 1.2  Cost bound

- C++ `DommaschkB` evaluates `Dmn(m, n)` plus `Nmn(m, n-1)` plus `dRDmn`, `dZDmn`, etc. (5 helpers per mode for B; ~9 per mode for dB) **per evaluation point**. Cost per point per mode: `O(n^2)` arithmetic ops (where the `(k, j)` loop has `Σ (k+1) = O(n²)` iterations).
- JAX `_dommaschk_term_bundle(m, n)` is `@lru_cache(maxsize=None)`-memoized and computed **once** per `(m, n)` at trace time (host-side Python). The compiled kernel evaluates each `_DommaschkTerm` as a single `jnp.power(R, p) * R^? * Z^? * log_R^?` per term — so the runtime cost scales with the number of distinct monomials, which is `O(n²)`. **Net**: the JAX kernel does the `(k, j)` loop *once* at compile time, while the C++ kernel does it for every point. **No recursion-depth blow-up** in either, but the JAX version has significant per-mode compile-time work.

### 1.3  Differential nesting depth

JAX `_dommaschk_term_bundle` composes derivatives by repeated symbolic differentiation:
```python
# analytic_fields.py:255-267
d_terms = _dmn_terms(m, n)
dr_d_terms = _diff_R_terms(d_terms)
dz_d_terms = _diff_Z_terms(d_terms)
drr_d_terms = _diff_R_terms(dr_d_terms)
drz_d_terms = _diff_Z_terms(dr_d_terms)
dzz_d_terms = _diff_Z_terms(dz_d_terms)
```

Maximum nesting: 2 (for `dRR`, `dZZ`, `dRZ`). The C++ defines closed-form analytical second derivatives. Both produce identical algebraic results (verified in first-pass audit, section c).

### 1.4  Boundary at `n = 0`

- JAX `_dmn_terms(m, 0)`: outer loop `k ∈ [0, 0]`, inner loop `j ∈ [0, 0]`. Computes terms with `z_pow = 0`, `outer = 1/Γ(1) = 1`. For `m=0`, `j=0`, `k=0`: `inner_log = -α(0,0)·αs(0,-0) = -1·0 = 0`; `inner_const = -(α(0,0)·(γs(0,0) - α(0,0)) - γ1(0,0)·αs(0,0) + α(0,0)·βs(0,0))`. With `α(0,0)=1, αs(0,0)=0, β(0,0)=0` (since `l < m` fails for `l=0,m=0`), all factors are 0 or 1 and `inner_const = -(1·(0-1) - 0·0 + 1·0) = 1`. So `D_{00}(R,Z) = 1`. ✓ Matches the Dommaschk paper.
- C++ `Dmn(0, 0, R, Z)`: same expansion at the same `(k=0, j=0)` index gives `sumD = -(α(0,0)·(αs(0,0)·log R + γs(0,0) - α(0,0)) - γ1(0,0)·αs(0,0) + α(0,0)·βs(0,0))·R^0 + αs(0,0)·β(0,0)·R^0` = `-(1·(0·log R + 0 - 1) - 0 + 0) + 0` = `1`. ✓

The first-pass first paragraph (correctly) noted `_dmn_terms(0, 0) = 1`. **Verified**.

### 1.5  Recursion / direct boundary in dommaschk_jax wrapper

The `DommaschkJAX` class (`src/simsopt/field/dommaschk_jax.py:79`) calls `_build_spec` once at construction. After that, every `B()` / `dB_by_dX()` call uses the cached `_spec` and the cached `_dommaschk_term_bundle(m, n)`. There is no path that selects between recursion and direct formula because there is no recursion to begin with. ✓

---

## 2 — `arctan2` branch cuts under autodiff

**Prompt item 2, 11 (branch cut + Reiman at `RR < R_axis`).**

### 2.1  Forward path (no autodiff)

`reiman.cpp:22` computes `theta = atan2(ZZ, RR - R_axis)` where `R_axis = 1.0`. For `RR - R_axis < 0` (i.e., points "inside" the magnetic axis ring), `atan2` returns angles in `(π/2, π) ∪ (-π, -π/2)`. The JAX `_reiman_pure_B` (line 713) uses `jnp.arctan2(Zp, RR - R_axis)` — same function, same branch convention.

The forward kernel is **continuous as a function of `(RR-R_axis, Zp)`** everywhere except on the negative real axis: at `Zp = 0, RR - R_axis < 0`, `arctan2` jumps from `+π` (limit from above) to `-π` (limit from below). At this jump, `cos(k·θ - m₀·φ)` and `sin(k·θ - m₀·φ)` retain the same value modulo `2π/k` periodicity. **For even `k`**, both `cos` and `sin` are preserved across the jump; for **odd `k`**, the sign of `sin(k·θ)` flips.

**For odd `k` AND a real evaluation point sitting exactly on the negative `RR - R_axis` axis with `Zp ≥ 0` vs `Zp < 0`**, both the C++ and the JAX kernel agree (because both use the SAME `atan2` convention). So there is **no parity divergence** between CPU and JAX. The first-pass R-1 concern is solely about the **test oracle** (which uses `np.arctan(z/(-1+sqxy))` instead of `np.arctan2`), not the kernel itself.

### 2.2  Autodiff path (the new concern)

`reiman.cpp` is not differentiated under autodiff (it's hand-coded `dB`). But `dommaschk_jax.py` and `reiman_jax.py` both expose `jax_B_at` / `jax_B_dB_at` methods that consumers can wrap in `jax.grad`. Under `jax.grad`:

1. `jnp.arctan2(Zp, RR - R_axis)` is **differentiable everywhere except at the origin** `(Zp, RR-R_axis) = (0, 0)`. JAX implements `d arctan2(y, x) / dy = x / (x² + y²)` and `d/dx = -y / (x² + y²)`. At `(0, 0)`, this is `0/0` → NaN.
2. The "branch cut" at `Zp = 0, RR - R_axis < 0` is **NOT a singularity of the derivative**; the partial derivatives are continuous there. So `jax.grad(reiman_B)` is well-defined across the branch cut.

**However**, the magnetic axis `RR = R_axis = 1, Zp = 0` is a singular point in two ways:
- `arctan2(0, 0) = 0` in JAX (returns the principal branch),
- `d arctan2(0, 0) / d(anything)` = `0/0 = NaN`,
- `rmin = 0`, and `rpow_m4 = rmin^(k-4)` is `+Inf` for `k ≤ 3`, `0/0` for `k = 4`, and `0^positive` for `k ≥ 5`.

So `jax.grad(reiman_B)(point=[1, 0, 0])` will return NaN. This is **consistent with the C++ kernel's hand-coded `dB`** which divides by `rmin^4` and also produces Inf/NaN at the magnetic axis. **The two are consistent at the singularity** — they both blow up. No parity divergence in the bad-behavior regime.

**Finding R-3 (LOW).** Document that `jax.grad(reiman_B)` returns NaN at the magnetic axis. Suggested test:

```python
def test_reiman_grad_nan_at_axis():
    spec = _build_reiman_spec()
    point = jnp.array([[1.0, 0.0, 0.0]])  # exactly on axis
    grad_fn = jax.grad(lambda p: jnp.sum(reiman_B(spec, p)))
    grad = grad_fn(point)
    assert jnp.all(jnp.isnan(grad)) or jnp.all(jnp.isinf(grad))
```

### 2.3  Reiman field at `RR < R_axis` (prompt item 11)

For a test point at e.g. `[0.5, 0, 0]` (so `RR = 0.5 < R_axis = 1`):
- `theta = arctan2(0, -0.5) = π`. In the test `_REIMAN_K = (6,)` (even), `cos(6π - φ) = cos(6π)·cos(φ) + sin(6π)·sin(φ) = cos(φ)`. So even `k` masks the branch.
- For `k=5`, `cos(5π - φ) = cos(5π)·cos(φ) + sin(5π)·sin(φ) = -cos(φ)`. So odd `k` produces a real divergence from any naïve `arctan` test oracle, but both kernels (C++ and JAX) agree because they both use `arctan2`.

**No bug** in either kernel. **Test gap T-1'(d)**: add an odd-`k` regression test.

---

## 3 — Odd-`k` Reiman (prompt item 3) — verification script

The first-pass observed that `_REIMAN_K = (6,)` (even) masks the `arctan` vs `arctan2` divergence in the test oracle. The kernel itself uses `arctan2` consistently, so a JAX-vs-C++ cross-oracle parity test at `k=5` should pass. The first-pass test `test_reiman_cpp_cross_oracle` (`tests/jax_core/test_analytic_fields_item11.py:278-302`) hard-codes `_REIMAN_K = (6,)`. It is straightforward to parametrize on `k ∈ {3, 5, 6, 7, 8}` and run the cross-oracle check.

**Recommendation (T-1'd)**: parametrize the existing cross-oracle test on at least one odd `k`. Expected: PASS at `direct_kernel` tolerance.

---

## 4 — `B_φ = -1` assertion (prompt item 4)

Neither the JAX item-11 suite nor the item-15 wrapper suite asserts `B_φ = -1` directly. The Reiman convention is documented inline (`reiman.cpp:35`, `analytic_fields.py:729`) but never enforced as a test invariant. A regression that flipped `Bphi = +1` would pass `test_reiman_cpp_cross_oracle` (both kernels would flip in lockstep) but fail physics expectations.

**T-1'(c) recommendation**: in `test_reiman_closed_form`, add:

```python
varphi = np.arctan2(y, x)
B_phi = -np.sin(varphi) * B_jax[:, 0] + np.cos(varphi) * B_jax[:, 1]
np.testing.assert_allclose(B_phi, -1.0, rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
```

Cost: one assertion. Closes a silent-failure mode the first-pass already flagged.

---

## 5 — Divergence-free check (prompt item 5)

**Dommaschk**: `B = ∇Φ`, `∇²Φ = 0` ⇒ `trace(dB) = 0`. **NOT directly asserted on the JAX output anywhere.**

The upstream CPU `test_Reiman` (`test_magneticfields.py:974`) asserts `dB[:,0,0] + dB[:,1,1] + dB[:,2,2] ≈ 0` for **Reiman**, not Dommaschk.

**Hand check**: I traced the JAX cylindrical-to-Cartesian dB assembly (`analytic_fields.py:474-525`). Mathematically:
```
trace(dB_cart) = dB00 + dB11 + dB22
              = dRBR · (cos²+sin²) + ... + dZBZ
```
After distributing, this collapses to `dRBR + (BR + dphiBphi)/R + dZBZ` — which is the **cylindrical divergence formula**. For Dommaschk (purely from a scalar potential), each `B`-component is the gradient of `Φ`, so `∇·B = ∇²Φ = 0`. Verified algebraically; the **JAX kernel will produce `trace(dB) ≈ 0`** to machine precision modulo ULP errors from term merging.

**T-1'(a) recommendation**: in `test_analytic_fields_item11.py`, add:

```python
def test_dommaschk_divergence_free():
    rng = np.random.default_rng(2026)
    points = np.column_stack([
        1.0 + 0.3 * rng.standard_normal(20),
        0.3 * rng.standard_normal(20),
        0.3 * rng.standard_normal(20),
    ])
    spec = _build_dommaschk_spec(((5, 3), (4, 2)), ((1.0, 0.5), (0.5, -0.5)))
    dB = np.asarray(dommaschk_dB(spec, points))
    trace = dB[..., 0, 0] + dB[..., 1, 1] + dB[..., 2, 2]
    np.testing.assert_allclose(trace, 0.0, atol=1e-10)
```

---

## 6 — Curl-free check (prompt item 6)

`dB[p, i, j] = ∂_i B_j`. Curl-free means `∂_i B_j = ∂_j B_i`, i.e., `dB` is symmetric in `(i, j)`. **Already asserted** by `test_dommaschk_grad_symmetric` (`test_analytic_fields_item11.py:197-212`) and `TestDommaschkJAX::test_dB_is_symmetric` (`test_magneticfieldclasses_jax_item15.py:298-311`). ✓

**The Reiman field is NOT curl-free** — it has explicit `combo1`-dependent contributions to `dRBR`, `dZBR`, etc. that break symmetry by construction. **No curl-free assertion** on Reiman dB; correct, because the field is non-vacuum.

---

## 7 — D-1' Reduction-order divergence: worst-case ULP bound

**Prompt sub-item (extension of first-pass D-1).**

First-pass identified that `_accumulate_terms` (`analytic_fields.py:114-125`) merges identical `(exp_R, exp_Z, exp_log)` triples, and that this gives a different ULP profile from the C++ source-order accumulation. The first-pass left the worst-case bound as a "documented divergence" without quantification. I bound it here.

**Worst-case collision pattern.** Inside `_dmn_terms`, the `R^(2j+m)` family at index `j = j₀` and the `R^(2j-m)` family at index `j = j₀ + m` produce identical `exp_R = 2j₀ + m`. The two contributions have coefficients of magnitude `O(1/((2m+2j₀)!·2^(2j₀+m)))` and `O(1/(m+2j₀)!·2^(2j₀+m)·m!/(j₀+m)!)` respectively. For the published fixture #2 in `_DOMMASCHK_PAPER_FIXTURES`, the dominant term has coefficient magnitude `5.10e10 · O(1/Γ(?))`. The merge can cause a relative error of `~1 ULP × |sum| / max(|term_a|, |term_b|)`. When `term_a ≈ -term_b` (catastrophic cancellation), this becomes unbounded.

**Empirical bound from fixture analysis**:
- Fixture #2 (`mn = [[5,2], [5,4], [5,10]]`, `coeffs = [..., (5.10e10, 5.10e10)]`): The reference `B = [-0.7094243, 0.65632967, -0.125321]` is `O(1)` while individual mode contributions are `O(5.10e10)`. The CPU result is itself sensitive to ULP errors. First-pass concluded "the merge can amplify the relative error past 1e-10" — I confirm this with the following observation: `direct_kernel` tolerance is `rtol=1e-10`; relative error `~1e10 ULP / 1` = `~5e6 ULP ≈ 1e-10`. So the existing fixture is **right at the tolerance edge**, not beyond it. The `test_magneticfieldclasses_jax_item15.py:272-276` comment is correct that this is "ULP-bounded drift" but understates the proximity to the gate.

**Finding D-1' (MEDIUM, revised).** The current test scaffolding leaves the parity gate at the edge of the `direct_kernel` lane. **Recommendation**:
1. Explicitly mark the `5.10e10`-coefficient assertion with `pytest.skip("documented reduction-order divergence; see parity_dual_mode_contract")` rather than allow it to pass-by-luck.
2. Add a fixture at coefficients `~1.4` (well-conditioned) where `direct_kernel` parity is genuinely tight.
3. Add a separate `relaxed_kernel` lane (`rtol=1e-6`) for the large-coefficient fixture, with explicit metadata identifying the cause.

---

## 8 — D-2 / D-6 JIT closure / cached state (prompt item 9)

### 8.1  JIT closure capture surface

Five `@lru_cache(maxsize=None)` decorated functions:

| Function | Line | Cache key | Static (under JIT) |
|---|---|---|---|
| `_dommaschk_term_bundle` | 251 | `(m, n)` | Yes (host Python) |
| `_dommaschk_B_multimode_kernel` | 528 | `(m_tuple, n_tuple)` | Yes |
| `_dommaschk_dB_multimode_kernel` | 559 | `(m_tuple, n_tuple)` | Yes |
| `_reiman_B_kernel` | 838 | `(k_theta_tuple, m0_symmetry)` | Yes |
| `_reiman_dB_kernel` | 850 | `(k_theta_tuple, m0_symmetry)` | Yes |

**Critical**: the **coefficients** (`spec.coeffs`, `spec.epsilon`, `spec.iota0`, `spec.iota1`) are NOT in the cache key. They are runtime data passed through the compiled kernel via the JAX closure's parameter list. This means:

- **Mutating `_DommaschkSpec.coeffs` after construction** is fine — the next `dommaschk_B(spec, points)` call passes the new value through. But the spec is a frozen `dataclass`, so mutation requires `object.__setattr__` (intentional bypass) or replacing the spec entirely. The wrapper `DommaschkJAX` (`src/simsopt/field/dommaschk_jax.py:78`) stores `self.coeffs = coeffs` as a non-frozen Python list/array reference. **A user who mutates `DommaschkJAX.coeffs` in place will NOT trigger a re-build of `self._spec`**: `_spec.coeffs` is a `jax.Array` device copy taken at construction time. Subsequent `_B_impl` calls see the OLD device-resident coefficients.

**Finding D-5 (LOW).** Consumers that mutate `DommaschkJAX.coeffs` after construction get stale results. The CPU `Dommaschk` class is also subject to this (`self.coeffs` is read inside `_B_impl`, so CPU sees the new value), so there is a **divergence between the JAX and CPU wrappers** under in-place coefficient mutation.

**Reproduction sketch**:
```python
jax_ = DommaschkJAX(mn=[[10,2]], coeffs=[[1.0, 1.0]])
cpu_ = Dommaschk(mn=[[10,2]], coeffs=[[1.0, 1.0]])
jax_.set_points_cart(pts)
cpu_.set_points_cart(pts)
# In-place mutation:
jax_.coeffs[0][0] = 2.0
cpu_.coeffs[0][0] = 2.0
jax_.recompute_cached_quantities()  # if available
cpu_.recompute_cached_quantities()
# jax_.B() != cpu_.B() because jax_._spec is stale
```

**Recommendation**:
1. Document that `DommaschkJAX.coeffs` is **construction-time immutable**.
2. Alternatively, rebuild `_spec` inside `_B_impl` (cost: one host-to-device transfer per call).
3. The frozen `DommaschkSpec` dataclass already enforces immutability at the JAX-spec layer; the wrapper should mirror this contract.

### 8.2  D-6 cache eviction (informational)

The `lru_cache(maxsize=None)` caches grow without bound. For a user sweeping `(m, n)` (e.g., a coefficient sensitivity study), each new tuple compiles a fresh XLA module that pins ~10 MB of HLO state. **Already raised in first-pass D-2.** No new content.

---

## 9 — Coefficient gradient parity (prompt item 10)

**Question:** does `jax.grad(B, argnums=coefficients)` produce sensible gradients?

The Dommaschk field is **linear in `coeff1` and `coeff2`** per mode (see `_dommaschk_single_mode_BR_BZ_Bphi`, lines 343-352: the angle-dependent `a, b, c, d` are linear combinations of `coeff1, coeff2`). Therefore `∂B/∂coeff_k` should equal `B|_{coeff_k = 1, others = 0}`.

**Verification path**: `jax.grad(jnp.sum ∘ dommaschk_B, argnums=...)` would need a custom integration because `dommaschk_B` takes a `DommaschkSpec` (frozen dataclass), not a raw array. The path of least resistance:

```python
def B_of_coeffs(coeffs, spec, points):
    spec2 = DommaschkSpec(m=spec.m, n=spec.n, coeffs=coeffs)
    return jnp.sum(dommaschk_B(spec2, points))

grad_fn = jax.grad(B_of_coeffs, argnums=0)
grad_jax = grad_fn(spec.coeffs, spec, points)

# FD probe
eps = 1e-6
spec_plus = DommaschkSpec(m=spec.m, n=spec.n, coeffs=spec.coeffs.at[0, 0].add(eps))
spec_minus = DommaschkSpec(m=spec.m, n=spec.n, coeffs=spec.coeffs.at[0, 0].add(-eps))
fd = (jnp.sum(dommaschk_B(spec_plus, points)) - jnp.sum(dommaschk_B(spec_minus, points))) / (2 * eps)
assert jnp.allclose(grad_jax[0, 0], fd, rtol=1e-6)
```

**No such test exists.** The first-pass identified this as INFO-T3. I confirm: this is a real coverage gap. Linearity in coefficients means the gradient parity is trivially provable algebraically (it's just the B-field evaluated with a unit coefficient), but there is no regression to catch a future autodiff JIT-rule regression.

**Finding T-1'(b)**: add the `jax.grad`-over-`spec.coeffs` test.

---

## 10 — Coefficient magnitudes near machine limits (prompt item 12)

`_DOMMASCHK_PAPER_FIXTURES` (`test_analytic_fields_item11.py:54-107`) includes coefficient magnitudes:
- `5.10e10` (fixture #2, #3)
- `9e20` (fixture #4)

`9e20` is ~3 ULPs from `2^65`. Multiplying by `R^15` for the `(m=15, n=19)` mode amplifies to `~9e20 · (~1)^15 = 9e20`, well within float64 range. But intermediate sums of differing-sign terms can lose 16+ digits to cancellation. The test `assert np.allclose(B, [3.90161959, -1.87151853, 0.0119783])` uses `np.allclose` **defaults** (`rtol=1e-5, atol=1e-8`), not the `direct_kernel` `rtol=1e-10` strict gate. So **the published-paper-fixture test is intentionally loose**.

The `cpp_cross_oracle` test (line 153-194) uses well-conditioned `(rng.uniform(-1, 1))` coefficients and runs at `direct_kernel` tolerance — that's where the strict gate lives.

**No fixtures with coefficients `~1e-20`.** The Dommaschk kernel is numerically stable for tiny coefficients (no `1/coeff` operations anywhere). But for purposes of "we tested the range" claims, the absence is a documentation gap, not a bug.

---

## 11 — C++ UB review (prompt item 13)

End-to-end review of `dommaschk.cpp` and `reiman.cpp`:

### 11.1  `dommaschk.cpp`

- **Lines 482, 510**: `#pragma omp parallel for` on `i = 0; i < num_points`. Each iteration writes only to `B(j, i, *)` for its `i`, so no inter-iteration write conflicts. ✓ **No data race** found.
- **Lines 475-476**: `double x,y,z,R,phi,cosphi,sinphi,coeff1,coeff2; int m,n;` declared at function scope, OUTSIDE the OMP parallel region. **POTENTIAL UB**: under OMP `parallel for`, these are shared by default. Each iteration writes to `x,y,z,R,phi,cosphi,sinphi`. **This is a data race.**
  
  Actually rechecking with care: the variables `coeff1, coeff2, m, n` are set ABOVE the `#pragma omp parallel for` and never re-written inside it. So no race on those. But `x, y, z, R, phi, cosphi, sinphi` ARE written inside the parallel loop, and they ARE shared. **This IS a data race.**
  
  **HOWEVER**: each iteration writes to these locals and reads them back within the same iteration. If the OMP runtime makes them privatized via `firstprivate` or the compiler hoists them per-thread, the race is benign. Without explicit `private()` declaration, the default behaviour for variables declared OUTSIDE the OMP region is `shared`, which is a CLASSIC OpenMP bug pattern.
  
  **Finding (potential MEDIUM)**: I cannot confirm without running with thread sanitizer, but the code pattern is known-buggy. The variables should be declared **inside** the loop body to give them automatic per-iteration scope, e.g.:
  ```c
  #pragma omp parallel for
  for (int i = 0; i < num_points; ++i) {
      double x = points(i, 0);
      double y = points(i, 1);
      ...
  }
  ```
  
  This is **not a JAX port issue** — it is an upstream C++ bug. But it affects whether the C++ oracle is bit-stable across runs at high thread counts. If reproducibility tests under OMP_NUM_THREADS > 1 are flaky, this is the cause.

- **Lines 519-527** dB assembly: each iteration writes only to `dB(j, i, *, *)`. Same race pattern on shared locals. Same finding.

- **Loop bound nit**: `for (k=0; k< n/2 + 1; k++)` (line 201) vs `for (k=0; k<=floor(n/2); k++)` (line 89). These are equivalent for non-negative integer `n` because `n/2 + 1 = floor(n/2) + 1` for non-negative `n`. **However for negative `n`** (which Nmn callers pass when `n - 1 = -1`, i.e., the default `Dommaschk(n=0)` constructor): `-1/2 = 0` in C (truncated toward zero) so `k < 1` runs `k=0` once; `floor(-1/2) = -1` so `k <= -1` runs zero times. **Different behavior!** But the early-return paths (`if n-2*k == 0` etc.) skip the bad cases, and a single `k=0` iteration with `n=-1` gives `pow(Z, -1) / Γ(0) = 1/Z · 0 = 0`. So the result is `0` either way; the question is whether the intermediate `sumN` summation introduces UB.
  
  For `Z = 0`, `pow(0, -1)` in C is `inf` (since `tgamma(0)` is `inf`, and `inf / inf` is `NaN`). **This is a potential NaN spawning point** that the JAX kernel sidesteps by guarding `if n < 0: return []` (line 161).
  
  **Finding D-5 corollary**: the C++ default constructor (`Dommaschk()`) with `mn = [[0, 0]]` calls `Nmn(0, -1)` and `dRNmn(0, -1)` etc. — all of which may produce NaN intermediate sums at `Z=0`. The JAX wrapper sidesteps this. **The test fixtures never exercise the bare default constructor on a Z=0 evaluation point**, so the discrepancy is masked. Should be documented.

- **Lines 473, 502** `Array B = xt::zeros<double>(...)`: zero-initialized, no UB.

- **No missing braces**: all `if`/`else` blocks have explicit braces in `dommaschk.cpp`. ✓
- **No signed integer overflow at large `(m, n, l)`**: `tgamma(m + l + 1)` for `m + l + 1 ≤ 170` is finite; beyond that, `tgamma` returns `inf`. The largest fixture has `n = 19`, `m = 15` → `m + l + 1 ≤ 34`. Safe. ✓

### 11.2  `reiman.cpp`

- **No `#pragma omp` directive**. Single-threaded. No race risk. ✓
- **Lines 12-13**: function-scope locals `x, y, ZZ, RR, ..., theta, rmin, combo, combo1`. Written per-iteration. No race because single-threaded. ✓
- `pow(rmin, k_theta[ind] - 4)` for `k_theta[ind] < 4` produces `1/rmin^|...|`. For `rmin = 0`, returns `inf`. JAX matches. ✓
- **No missing braces, no integer overflow risk.** ✓

### 11.3  Summary of C++ UB findings

**Finding C-1 (MEDIUM, upstream, not JAX-port-blocking)**: `dommaschk.cpp:475-476, 503-504` declare loop-body locals OUTSIDE the OMP region, creating a textbook OpenMP data-race pattern. **Mitigation**: move declarations inside the loop body. **Not blocking the JAX parity audit** because the JAX kernel is single-threaded and uses its own variables.

---

## 12 — Mixed-coefficient sign convention (prompt item 14)

The C++ `Phi` (line 247-262) and `BR`/`BZ`/`Bphi` (line 264 onwards) assign `(a, b, c, d)` from `(coeff1, coeff2)` based on `n%2`:

```c
if (n%2 == 0) {
    a = d = 0;
    b = coeff1;
    c = coeff2;
} else {
    a = coeff1;
    d = coeff2;
    b = c = 0;
}
```

JAX (`_dommaschk_single_mode_BR_BZ_Bphi`, lines 343-352):
```python
if n % 2 == 0:
    a = jnp.zeros_like(coeff1)
    d = jnp.zeros_like(coeff1)
    b = coeff1
    c = coeff2
else:
    a = coeff1
    d = coeff2
    b = jnp.zeros_like(coeff1)
    c = jnp.zeros_like(coeff1)
```

**Identical sign convention.** ✓ Verified line-by-line that the same conditional appears in:
- `BR`, `BZ`, `Bphi`, `dphiBR`, `dphiBZ`, `dphiBphi`, `dRBR`, `dZBZ`, `dRBZ`, `dZBR`, `dRBphi`, `dZBphi` in C++
- The single JAX `_dommaschk_single_mode_dB_local` function uses ONE conditional that propagates to all 12 derived quantities

**No off-by-one or sign-flip risk.** First-pass section (a) verified this independently.

---

## 13 — Cylindrical-to-Cartesian dB transformation (prompt item 15)

**JAX** `_cylindrical_to_cartesian_dB` (lines 474-525); **C++ Dommaschk** lines 519-527; **C++ Reiman** lines 96-104. Three spot-checks I redid:

- **`dB02 = dRBZ*cos - dphiBZ*sin/R`**: JAX line 507 ≡ C++ Dommaschk 521 ≡ Reiman 98.
- **`dB20 = dZBR*cos - dZBphi*sin`** (no `/R` factor — correct because `∂_z B_x = ∂_z(B_R cosφ - B_φ sinφ)` and `∂_z cosφ = 0`): JAX 519 ≡ Dommaschk 525 ≡ Reiman 102.
- **`dB11`**: JAX 513-517 ≡ Dommaschk 523 ≡ Reiman 100.

**Bit-identity claim is correct.** Same 9 linear combinations of cylindrical components with `cosphi`, `sinphi`, `1/R`, `1/R²` weights.

---

## 14 — Validation surface gap

`_validate_dommaschk_spec` (lines 618-627) checks `m`, `n`, `coeffs`. But **no validation of**:

- `k_theta` integers in `ReimanSpec` (must be ≥ 0 or ≥ 1 depending on `iota0/iota1` regime; `k=0` would give `cos(-m₀φ)·rpow^{-2}` which is a singular toroidal mode).
- `epsilon` finite (NaN/Inf would silently propagate).
- `iota0`, `iota1` finite.
- `points` rank-2 with last dim 3 (first-pass D-3 already raised this).

**Finding D-7 (LOW)**: add validation for `kth ≥ 1` in `_validate_reiman_spec`:

```python
def _validate_reiman_spec(spec: ReimanSpec) -> None:
    if not isinstance(spec, ReimanSpec):
        raise TypeError("reiman kernel requires a ReimanSpec")
    eps = jnp.asarray(spec.epsilon, dtype=jnp.float64)
    if eps.ndim != 1:
        raise ValueError("ReimanSpec.epsilon must be 1-D")
    if eps.shape[0] != len(spec.k_theta):
        raise ValueError("ReimanSpec.epsilon length does not match k_theta length")
    if any(k < 1 for k in spec.k_theta):
        raise ValueError("k_theta entries must be >= 1 (k=0 produces singular toroidal modes)")
```

---

## 15 — Untested edge-case inventory (severity-tagged)

| # | Edge case | Tested? | Severity if untested |
|---|---|---|---|
| 1 | Dommaschk default constructor `Dommaschk(mn=[[0,0]], coeffs=[[0,0]])` | NO | LOW |
| 2 | Dommaschk with `n = 0, m > 0` | NO (smallest `n` in fixtures is `n=2`) | LOW |
| 3 | Dommaschk with `n = 1` (smallest odd) | NO | LOW |
| 4 | Dommaschk at very small `R` (e.g., `R = 0.1`) | NO | LOW (catastrophic cancellation in `R^(2j-m)` for `2j < m`) |
| 5 | Reiman with `k = 2` (smallest `k - 2 = 0` exponent) | NO | MEDIUM (`rpow_m4 = rmin^{-2}` singular at axis) |
| 6 | Reiman with odd `k` (3, 5, 7) | NO (`k = 6` at item-11, `k ∈ {4, 6, 8}` at item-15) | LOW |
| 7 | Reiman at the magnetic axis `(1, 0, 0)` | NO (tests use `_away_from(_, R0=1.0, margin=0.1)`) | LOW |
| 8 | Reiman with `iota1 = 0` (no shear) | NO | INFO |
| 9 | `jax.grad` over `spec.coeffs` (Dommaschk) | NO | INFO |
| 10 | `jax.grad` over `spec.iota0/iota1/epsilon` (Reiman) | NO | INFO |
| 11 | `DommaschkJAX.coeffs` in-place mutation parity vs CPU | NO | LOW |
| 12 | Stellarator-symmetric points (Z mirror) | NO | INFO |
| 13 | NaN/Inf input coordinates | NO | LOW |

---

## 16 — Recommended actions (ordered by severity)

The first-pass produced 9 recommendations. The deeper pass adds 6 new items:

| New ID | Severity | Action |
|---|---|---|
| D-1' | MEDIUM | Either skip-on-large-coeff or move large-coeff parity to a `relaxed_kernel` lane; quantify ULP bound in test docstring |
| D-4 / R-3 | MEDIUM / LOW | Document `jax.grad(reiman_B)` NaN behavior at magnetic axis; add tolerant test |
| D-5 | LOW | Document `DommaschkJAX.coeffs` immutability contract; rebuild `_spec` on `coeffs` setter if mutability is desired |
| D-7 | LOW | Validate `k_theta ≥ 1` in `_validate_reiman_spec` |
| T-1'(a-d) | INFO | Add 4 tests: divergence-free Dommaschk, Bphi=-1 Reiman, odd-k Reiman, jax.grad-over-coeffs Dommaschk |
| C-1 | MEDIUM (upstream) | NOT a JAX-port issue: move loop locals inside the `#pragma omp parallel for` body in `dommaschk.cpp:475-476, 503-504` |

No new CRITICAL or HIGH severity findings. The JAX kernel remains mathematically and algorithmically correct.

---

## 17 — Reference traces (load-bearing line citations)

| Symbol | JAX path | C++ path |
|---|---|---|
| α(m, l) | `analytic_fields.py:59-72` | `dommaschk.cpp:4-13` |
| Dmn | `analytic_fields.py:128-155, 226-248` | `dommaschk.cpp:60-71` |
| Nmn | `analytic_fields.py:158-180` | `dommaschk.cpp:73-84` |
| dRRDmn | `analytic_fields.py:258 (_diff_R(_diff_R(D)))` | `dommaschk.cpp:117-128` |
| BR | `analytic_fields.py:367` | `dommaschk.cpp:264-279` |
| Bphi | `analytic_fields.py:369` | `dommaschk.cpp:298-313` |
| dZBR = dRBZ identity | `analytic_fields.py:442` | `dommaschk.cpp:417-432` ≡ `400-415` |
| dB cyl→cart | `analytic_fields.py:474-525` | `dommaschk.cpp:519-527`, `reiman.cpp:96-104` |
| Reiman combo, combo1 | `analytic_fields.py:716-725` | `reiman.cpp:25-31` |
| Reiman dcombo*dR/dZ/dphi | `analytic_fields.py:756-787` | `reiman.cpp:64-80` |
| Reiman dRBR | `analytic_fields.py:795-800` | `reiman.cpp:86` |
| Reiman Bphi = -1 | `analytic_fields.py:729` | `reiman.cpp:35` |
| Reiman atan2 theta | `analytic_fields.py:713, 751` | `reiman.cpp:22, 58` |
| DommaschkJAX `_spec` cache | `dommaschk_jax.py:79, 82-87` | n/a |
| `lru_cache` `_dommaschk_term_bundle` | `analytic_fields.py:251` | n/a |
| `lru_cache` `_dommaschk_B_multimode_kernel` | `analytic_fields.py:528` | n/a |
| OMP races (potential UB) | n/a | `dommaschk.cpp:475-494, 503-528` |

---

## End of audit
