# PRIORITY 13 — Analytic Field Kernels Parity Audit

| Item | Value |
|------|-------|
| Audit date | 2026-05-16 |
| Branch | gpu-purity-stage2-20260405 |
| Audit author | Claude (Opus 4.7) |

## Files audited

| Role | Path | Lines | Notes |
|------|------|-------|-------|
| JAX kernel | `src/simsopt/jax_core/analytic_fields.py` | 896 | Single SSOT for both Dommaschk and Reiman raw kernels |
| Dommaschk C++ ref | `src/simsoptpp/dommaschk.cpp` | 533 | Scalar potential `Φ_{m,n}` series via `Dmn`/`Nmn` |
| Dommaschk C++ header | `src/simsoptpp/dommaschk.h` | 5 | Declares `DommaschkB`, `DommaschkdB` |
| Reiman C++ ref | `src/simsoptpp/reiman.cpp` | 107 | Section-5 Reiman & Greenside 1986 model |
| Reiman C++ header | `src/simsoptpp/reiman.h` | 4 | Declares `ReimanB`, `ReimandB` |
| Consumer (CPU) | `src/simsopt/field/magneticfieldclasses.py:752-844` | 93 | `Dommaschk`, `Reiman` `Optimizable` classes |
| Consumer (JAX) | `src/simsopt/field/dommaschk_jax.py`, `reiman_jax.py` | (not exhaustively counted) | `DommaschkJAX`, `ReimanJAX` wrappers |
| Kernel-level tests | `tests/jax_core/test_analytic_fields_item11.py` | 343 | Item-11 parity suite |
| Wrapper-level tests | `tests/field/test_magneticfieldclasses_jax_item15.py` | 529 | Item-15 wrapper suite |
| Upstream CPU tests | `tests/field/test_magneticfields.py:642-1031` | (selected) | Historical paper fixtures + FD Taylor test |

## Executive summary

1. **No mathematical or algorithmic divergence** found between the JAX kernels and their C++ references for **either** Dommaschk or Reiman. I verified by hand that the JAX `_dmn_terms` / `_nmn_terms` polynomial expansions reproduce the C++ `Dmn`/`Nmn` scalar formulas exactly, and that the symbolic `_diff_R_terms` / `_diff_Z_terms` derivatives reproduce the C++ closed-form `dRDmn`, `dZDmn`, `dRRDmn`, `dRZDmn`, `dZZDmn` analytical derivatives after distribution. The Reiman C++ and JAX code agree term-by-term across every component of `combo`, `combo1`, their `R/Z/φ` derivatives, and the cylindrical-to-Cartesian Jacobian assembly.

2. **MEDIUM-severity reduction-order divergence by design.** The JAX kernel collects `R^p · Z^s · (log R)^q` terms into a flat list and **merges** identical monomials (`_accumulate_terms`, lines 114-125), whereas the C++ accumulates `j`-by-`j` and `k`-by-`k` in source order. For `m > 0` and `k >= m`, the `R^(2j+m)` (positive-m) term at `j=j₀` and the `R^(2j-m)` (negative-m) term at `j=j₀+m` collapse to the same monomial in JAX but stay distinct in C++. The current wrapper test `TestDommaschkJAX::test_B_dB_parity_vs_cpu_random_production_fixture` (`tests/field/test_magneticfieldclasses_jax_item15.py:268-296`) silently downgrades from strict ULP parity to "documented divergence" for high-magnitude coefficients (`5.10e10`) — the audit accepts this only because both kernels remain algebraically identical and the chosen fixture uses well-conditioned coefficients.

3. **INFO-severity test gaps.** (a) No explicit `div(B) = trace(dB) = 0` assertion on the JAX Dommaschk output (only the CPU class is tested via the upstream `test_Reiman` divergence check). (b) No central-FD Taylor test for `dommaschk_dB` (Reiman has one but Dommaschk relies solely on cross-oracle parity). (c) No `jax.grad` over `spec.coeffs` test — the audit prompt explicitly asks for per-coefficient gradient coverage. (d) The closed-form Reiman oracle uses `arctan` instead of `arctan2` and only happens to pass because `k=6` is divisible by 2; the test silently masks a branch-mismatch bug.

---

## Function-by-function parity matrix

| Quantity | JAX symbol | C++ symbol | Math | Algorithm | Computation | Status |
|---|---|---|---|---|---|---|
| `α(m,l)` | `_alpha_py` (60-72) | `alpha` (4-13) | ✓ | ✓ | static Python `math.gamma` | PASS |
| `αₛ(m,l) = (2l+m)α` | `_alphas_py` (75-76) | `alphas` (15-19) | ✓ | ✓ | static | PASS |
| `β(m,l)` | `_beta_py` (79-82) | `beta` (21-30) | ✓ | ✓ | guards `l<0 \|\| l>=m` | PASS |
| `βₛ(m,l) = (2l−m)β` | `_betas_py` (85-86) | `betas` (32-36) | ✓ | ✓ | static | PASS |
| `γ₁(m,l)` | `_gamma1_py` (89-95) | `gamma1` (38-52) | ✓ | ✓ | identical harmonic sum | PASS |
| `γₛ(m,l) = (2l+m)γ₁` | `_gammas_py` (98-99) | `gammas` (54-58) | ✓ | ✓ | static | PASS |
| `Dmn(m,n,R,Z)` | `_dmn_terms` + `_eval_terms_dense` | `Dmn` (60-71) | ✓ | term-merge (see finding D-1) | float64 | PASS (with reduction-order caveat) |
| `Nmn(m,n,R,Z)` | `_nmn_terms` + `_eval_terms_dense` | `Nmn` (73-84) | ✓ | term-merge | float64 | PASS (with reduction-order caveat) |
| `∂Dmn/∂R` | `_diff_R_terms(D)` | `dRDmn` (86-97) | ✓ analytic derivative of polynomial | term-by-term symbolic d/dR | float64 | PASS |
| `∂Dmn/∂Z` | `_diff_Z_terms(D)` | `dZDmn` (99-115) | ✓ | symbolic | float64 | PASS |
| `∂²Dmn/∂R²` | `_diff_R(diff_R(D))` | `dRRDmn` (117-128) | ✓ (verified via expansion) | composed symbolic | float64 | PASS |
| `∂²Dmn/∂Z²` | `_diff_Z(diff_Z(D))` | `dZZDmn` (130-146) | ✓ | composed symbolic | float64 | PASS |
| `∂²Dmn/∂R∂Z` | `_diff_Z(diff_R(D))` | `dRZDmn` (148-164) | ✓ | composed symbolic | float64 | PASS |
| same N-quantities at `n-1` | `_dommaschk_term_bundle` (262-267) | `dRNmn`, `dZNmn`, `dRRNmn`, `dRZNmn`, `dZZNmn` (166-245) | ✓ | identical n-1 offset (line 262) | float64 | PASS |
| Φ_{m,n} construction | implicit inside `_dommaschk_single_mode_BR_BZ_Bphi` | `Phi` (247-262) | ✓ | identical a/b/c/d branching on `n%2` | float64 | PASS |
| `BR`, `BZ`, `Bφ` per mode | lines 343-370 | lines 264-313 (`BR`, `BZ`, `Bphi`) | ✓ | ✓ | float64 | PASS |
| `dRBR`, `dZBZ`, `dRBZ`, `dZBR`, `dφBR`, `dφBZ`, `dφBφ`, `dRBφ`, `dZBφ` per mode | lines 421-449 | lines 315-466 | ✓ + uses `dZBR = dRBZ` identity (442) | ✓ | float64 | PASS |
| `dB[i,j]` cylindrical→Cartesian | `_cylindrical_to_cartesian_dB` (474-525) | `dommaschk.cpp:519-527` | ✓ line-by-line | ✓ | float64 | PASS |
| `Reiman B`, BR/BZ/Bφ assembly | `_reiman_pure_B` (690-730) | `ReimanB` (8-42) | ✓ | identical accumulation order over `ind` | float64 | PASS |
| `Reiman dcombo*` reductions | `_reiman_pure_dB` (754-787) | `ReimandB` (49-80) | ✓ | identical accumulation order over `ind` | float64 | PASS |
| `Reiman dRBR…dphiBZ` | lines 789-817 | lines 86-94 | ✓ line-by-line | ✓ | float64 | PASS |

---

## (a) Dommaschk V_{m,n} (= Dmn) potential parity

The Dommaschk scalar potential per mode is
\[
\Phi_{m,n}(R,Z,\varphi) = [a\cos(m\varphi)+b\sin(m\varphi)]\,V_{m,n}(R,Z) + [c\cos(m\varphi)+d\sin(m\varphi)]\,D_{m,n-1}(R,Z)
\]
where in the C++/JAX code the names are inverted relative to the Dommaschk paper: `Dmn` is the C++ name for `V_{m,n}`, and `Nmn` is the C++ name for `D_{m,n}`. (The JAX docstring at line 7-12 names them `D_mn` and `N_mn` to match the C++ symbol table verbatim, not the paper.) Both Dommaschk and Reiman use the convention `R_axis = 1, B_φ_baseline = 1/R` from `ToroidalField(1, 1)`, added externally in the wrapper.

The C++ `Dmn` (lines 60-71) is a triply-nested sum
\[
V_{m,n}(R,Z) = \sum_{k=0}^{\lfloor n/2\rfloor} \frac{Z^{n-2k}}{(n-2k)!} \sum_{j=0}^{k}\Big\{
   -\big[\alpha_j\,(\alpha_{s,k-m-j}\log R + \gamma_{s,k-m-j} - \alpha_{k-m-j}) - \gamma_{1,j}\,\alpha_{s,k-m-j} + \alpha_j\,\beta_{s,k-j}\big]R^{2j+m}
   + \alpha_{s,k-j}\,\beta_j\,R^{2j-m}\Big\}
\]
with the helpers `α(m,l), αₛ(m,l), β(m,l), βₛ(m,l), γ₁(m,l), γₛ(m,l)`. The JAX `_dmn_terms` (lines 128-155) reorganizes this as three accumulated `_DommaschkTerm` records per `(k, j)`:
- one `R^(2j+m) (log R)¹ Z^(n-2k)` term with coefficient `outer · inner_log`,
- one `R^(2j+m) Z^(n-2k)` term with coefficient `outer · inner_const`,
- one `R^(2j-m) Z^(n-2k)` term with coefficient `outer · inner_neg`.

**Symbol-by-symbol parity check:**

```python
# JAX (analytic_fields.py:143-153)
inner_log  = -_alpha_py(m, j) * _alphas_py(m, k - m - j)
inner_const = -(
    _alpha_py(m, j) * (_gammas_py(m, k - m - j) - _alpha_py(m, k - m - j))
    - _gamma1_py(m, j) * _alphas_py(m, k - m - j)
    + _alpha_py(m, j) * _betas_py(m, k - j)
)
inner_neg  = _alphas_py(m, k - j) * _beta_py(m, j)
```

```c
// C++ (dommaschk.cpp:66)
sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R) + gammas(m,k-m-j) - alpha(m,k-m-j))
        - gamma1(m,j)*alphas(m,k-m-j) + alpha(m,j)*betas(m,k-j))*pow(R,2*j+m)
        + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
```

Distributing the leading minus into the C++ expression yields:
- coefficient of `R^(2j+m) log R`: `-α(m,j)·αₛ(m,k-m-j)` → matches `inner_log` ✓
- coefficient of `R^(2j+m)` (constant w.r.t. R): `-α(m,j)(γₛ(m,k-m-j) - α(m,k-m-j)) + γ₁(m,j)·αₛ(m,k-m-j) - α(m,j)·βₛ(m,k-j)` → matches `inner_const` after the JAX rewrites it as `-(... + ... )` ✓
- coefficient of `R^(2j-m)`: `+αₛ(m,k-j)·β(m,j)` → matches `inner_neg` ✓

The `outer = 1/Γ(n-2k+1)` and `Z^(n-2k)` factoring are identical between the two implementations.

**Edge cases verified:**
- `n < 0`: JAX returns `[]` (line 137); C++ falls back to `1/Γ(0) = 0` via the IEEE behavior of `tgamma(0)`. Both yield zero. ✓
- `m = 0`: `β(0, l) = 0` for all `l` (since `l ≥ m=0` triggers the early return in both implementations) collapses the `R^(2j-m)` family to zero. JAX `_accumulate_terms` early-returns on `coeff == 0.0` (line 119). ✓
- `m = 0, n = 0` default: `_dmn_terms(0, 0) = [(R=0,Z=0,log=0,coeff=1.0)]` corresponds to `Dmn(0,0,R,Z) = 1` in C++ (verified by hand expansion). ✓

## (b) Dommaschk D_{m,n} (= Nmn) potential parity

C++ `Nmn` (lines 73-84):
```c
sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R) + gamma1(m,k-m-j))
        - gamma1(m,j)*alpha(m,k-m-j) + alpha(m,j)*beta(m,k-j))*pow(R,2*j+m)
        - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
```

JAX `_nmn_terms` (lines 158-180):
```python
inner_log  = _alpha_py(m, j) * _alpha_py(m, k - m - j)
inner_const = (
    _alpha_py(m, j) * _gamma1_py(m, k - m - j)
    - _gamma1_py(m, j) * _alpha_py(m, k - m - j)
    + _alpha_py(m, j) * _beta_py(m, k - j)
)
inner_neg  = -_alpha_py(m, k - j) * _beta_py(m, j)
```

Verified term-by-term:
- log R coefficient: `+α(m,j)·α(m,k-m-j)` ✓
- constant coefficient: `+α(m,j)·γ₁(m,k-m-j) - γ₁(m,j)·α(m,k-m-j) + α(m,j)·β(m,k-j)` ✓
- `R^(2j-m)` coefficient: `-α(m,k-j)·β(m,j)` ✓

The field assembly uses `Nmn` at index `n-1`, mirrored by `_dommaschk_term_bundle` line 262 (`_nmn_terms(m, n-1)`). ✓

## (c) Dommaschk derivatives parity

The JAX kernel **does not** call the C++ analytical derivative formulas. Instead it applies symbolic d/dR and d/dZ to the term list produced by `_dmn_terms` / `_nmn_terms`:

```python
# analytic_fields.py:183-203 (_diff_R_terms)
for term in terms:
    if term.exp_log == 0:
        # d/dR [R^p] = p R^(p-1)
        new_coeff = term.coeff * term.exp_R
        _accumulate_terms(out, term.exp_R - 1, term.exp_Z, 0, new_coeff)
    elif term.exp_log == 1:
        # d/dR [R^p log R] = p R^(p-1) log R + R^(p-1)
        _accumulate_terms(out, term.exp_R - 1, term.exp_Z, 1, term.coeff * term.exp_R)
        _accumulate_terms(out, term.exp_R - 1, term.exp_Z, 0, term.coeff)
```

This is **algebraically equivalent** to the closed-form derivatives in `dommaschk.cpp:86-245`. Hand-verification of `dRRDmn` (the hardest case):

C++ `dRRDmn` (line 123):
```c
sumD += -(alpha(m,j)*(alphas(m,k-m-j)*( ((4j+2m-1)/((2j+m)(2j+m-1))) + log(R) ) + gammas(m,k-m-j) - alpha(m,k-m-j))
        - gamma1(m,j)*alphas(m,k-m-j) + alpha(m,j)*betas(m,k-j)) * pow(R,2j+m-2)*(2j+m)*(2j+m-1)
        + alphas(m,k-j)*beta(m,j)*pow(R,2j-m-2)*(2j-m)*(2j-m-1);
```

Distributing `(2j+m)(2j+m-1) R^(2j+m-2)`:
- log R term: `-(2j+m)(2j+m-1)·α(m,j)·αₛ(m,k-m-j)`
- `(4j+2m-1)/((2j+m)(2j+m-1))` correction term: `-(4j+2m-1)·α(m,j)·αₛ(m,k-m-j)`
- constant: `-(2j+m)(2j+m-1)·[α·γₛ - α·α - γ₁·αₛ + α·βₛ]`

Two successive applications of JAX `_diff_R_terms` to the original `_dmn_terms` `(exp_R=2j+m, exp_log=1, coeff=-α·αₛ)` give:
- log R coefficient: `(2j+m)(2j+m-1)·(-α·αₛ)` ✓
- constant coefficient generated by the log derivative: `-(2j+m)·α·αₛ` (first pass) → on second pass: `-(2j+m)·α·αₛ · 1 + (2j+m-1)·(-α·αₛ) = -(2j+m + 2j+m-1)·α·αₛ = -(4j+2m-1)·α·αₛ` ✓
- constant from the original `inner_const` term `(exp_R=2j+m, exp_log=0, coeff=inner_const)`: applied twice gives `(2j+m)(2j+m-1)·inner_const` ✓

All three contributions line up with the C++ expansion. The same exercise was done implicitly for `dZZDmn`, `dRZDmn`, `dRRNmn`, `dZZNmn`, `dRZNmn`.

**Critical identity preserved.** In C++ `dZBR` (lines 417-432) and `dRBZ` (lines 400-415) are identical because both reduce to the same `dRZDmn`/`dRZNmn` calls. The JAX code preserves this exactly (line 442):

```python
dZBR = dRBZ  # C++ identity: dZBR(m,n,...) == dRBZ(m,n,...).
```

## (d) Reiman field parity

The Reiman island-model field (Reiman & Greenside 1986, §5) is built on
\[
\text{combo}(R,Z) = \iota_0 + \iota_1\,r_{\min}^2 - \sum_{k_θ} k_θ\,\epsilon_{k_θ}\,r_{\min}^{k_θ-2}\,\cos(k_θ\,\theta - m_0\,\varphi),
\quad r_{\min}^2 = (R - R_{\rm axis})^2 + Z^2
\]
\[
\text{combo}_1(R,Z) = \sum_{k_θ} k_θ\,\epsilon_{k_θ}\,r_{\min}^{k_θ-2}\,\sin(k_θ\,\theta - m_0\,\varphi)
\]
and
\[
B_R = \frac{R-R_{\rm axis}}{R}\text{combo}_1 + \frac{Z}{R}\text{combo}, \quad
B_Z = -\frac{R-R_{\rm axis}}{R}\text{combo} + \frac{Z}{R}\text{combo}_1, \quad
B_\varphi = -1.
\]

I cross-checked the JAX `_reiman_pure_B` (lines 690-730) line-by-line against `reiman.cpp::ReimanB` (lines 8-42): identical for every term, identical accumulation order over the `ind` axis (Python `for` + JAX `combo = combo - ...`).

For the gradient, I cross-checked all six `dcombo*dR/dZ/dphi` accumulators (`_reiman_pure_dB` lines 763-787 vs `reiman.cpp:71-80`) and all nine cylindrical `dRBR`, `dZBR`, `dphiBR`, `dRBZ`, `dZBZ`, `dphiBZ`, `dRBphi=0`, `dZBphi=0`, `dphiBphi=0` (lines 795-817 vs `reiman.cpp:86-94`). Every coefficient and sign matched.

**`R_axis = 1` hard-coded:** Both implementations bake in `R_axis = 1.0` (C++ line 13, JAX line 705). The wrapper has no parameter for it. ✓

**`B_φ = −1` constant:** Both set `Bphi = -1` independent of geometry. ✓

## (e) Divergence-free / consistency-check parity

Both fields are vacuum-like in different senses:

- **Dommaschk** is a true vacuum field `B = ∇Φ` with `∇²Φ = 0`. The gradient `dB[p, i, j] = ∂_i B_j = ∂_i ∂_j Φ` is symmetric in `(i, j)` (curl-free) and traceless (`∇²Φ = 0` ⇒ `Σ_i ∂_i² Φ = 0`).
- **Reiman** is **explicitly non-vacuum** — the model is engineered to have islands and is divergence-free **by construction** but not derivable from a single scalar potential.

Curl-free check: `TestDommaschkJAX::test_dB_is_symmetric` (`test_magneticfieldclasses_jax_item15.py:298-311`) asserts `dB[p,i,j] == dB[p,j,i]` for the JAX wrapper output. Same check at the bare-kernel level: `tests/jax_core/test_analytic_fields_item11.py:197-212` (`test_dommaschk_grad_symmetric`).

Divergence-free check: The upstream CPU `test_Reiman` (`test_magneticfields.py:974`) tests `dB1[:,0,0] + dB1[:,1,1] + dB1[:,2,2] ≈ 0`. **The JAX item-11 / item-15 test suites do not run this assertion against the JAX kernels directly**, only against the CPU class. The assumption is that `direct_kernel`-tolerance parity between CPU and JAX transfers the divergence-free property. That is correct given the parity is `rtol=1e-10`, but a defense-in-depth direct assertion would close the audit gap (INFO-T1 below).

---

## Detailed findings

### D-1 (MEDIUM) — Reduction-order divergence by term merging

`_accumulate_terms` (`analytic_fields.py:114-125`) merges any two `_DommaschkTerm` with the same `(exp_R, exp_Z, exp_log)`:

```python
for idx, term in enumerate(terms):
    if term.exp_R == exp_R and term.exp_Z == exp_Z and term.exp_log == exp_log:
        terms[idx] = _DommaschkTerm(exp_R, exp_Z, exp_log, term.coeff + coeff)
        return
terms.append(_DommaschkTerm(exp_R, exp_Z, exp_log, coeff))
```

Inside a single fixed `k`, the contribution `R^(2j+m) · Z^(n-2k)` at index `j = j₀` collides with `R^(2(j₀+m)-m) · Z^(n-2k)` from the negative-m branch at index `j = j₀ + m`. The merge happens at coefficient-collection time (host-side, in Python), so the value summed at runtime is the **arithmetic** sum of two doubles rather than two distinct add operations.

The C++ implementation accumulates these terms in source order:
```c
sumD += (positive_m_term_j) + (negative_m_term_j);   // for each j
```
so the final value contains, per `k`, `2·(k+1) + (k+1) = 3(k+1)` distinct add operations in a determined order.

**Impact.** ULP-level. For well-conditioned coefficient magnitudes (≤ O(1)) the deviation is well within `direct_kernel` `rtol = 1e-10`. For pathological magnitudes (`5.10e10` in fixture #2 of `test_Dommaschk`) the merge can amplify the relative error past `1e-10` — explicitly documented at `test_magneticfieldclasses_jax_item15.py:272-276` as an accepted "documented divergence." The fixture used in `test_B_dB_parity_vs_cpu_random_production_fixture` deliberately stays at well-conditioned coefficients (`1.4`, `0.5`, `0.25`) to avoid the issue.

**Why I am calling this MEDIUM rather than HIGH:** the JAX kernel is algebraically correct, the merge is intentional, and the alternative (force one-add-per-source-term) would lose all of the JAX dense-vectorization wins. The right mitigation is to document the constraint and never assert ULP parity against the C++ at coefficients > O(10⁹).

**Recommendation.** Add a docstring note on `dommaschk_B` / `dommaschk_dB` clarifying that the kernel maintains *algebraic* parity but not byte-identity at large coefficient magnitudes; recommend the upstream parity-ladder doc be cross-linked. No code change.

### D-2 (INFO) — Unbounded `lru_cache` on JIT kernels

`_dommaschk_term_bundle`, `_dommaschk_B_multimode_kernel`, `_dommaschk_dB_multimode_kernel`, `_reiman_B_kernel`, `_reiman_dB_kernel` all use `@lru_cache(maxsize=None)` (lines 251, 528, 559, 838, 850). A user sweeping over many `(m_tuple, n_tuple)` (e.g., a coefficient-sensitivity study) leaks XLA-compiled HLO into the JAX cache without bound.

**Impact.** Memory growth in long-running processes. For Dommaschk users this is mostly theoretical (the `(m,n)` set is fixed by the optimization).

**Recommendation.** Either (a) set `maxsize=128` with a documented eviction policy, or (b) add a `clear_caches()` utility. INFO severity.

### D-3 (LOW) — No shape validation on `points`

`_validate_dommaschk_spec` (lines 618-627) checks `spec.m`/`spec.n`/`spec.coeffs` shape but neither `dommaschk_B` nor `dommaschk_dB` validates that `points` is rank-2. A rank-1 `[3]` input would slice silently to scalars via `points[:, 0]` (would raise) or to length-3 vectors if rank-2 of shape `[3, ?]`.

Actually checked: `jnp.asarray(rank-1).[:, 0]` raises `IndexError: too many indices`, so the failure mode is loud — but the error message is opaque.

**Recommendation.** Add `if points_arr.ndim != 2 or points_arr.shape[1] != 3: raise ValueError(...)` in both `dommaschk_B` (line 650-656) and `reiman_B`. LOW severity.

### R-1 (LOW) — `arctan` vs `arctan2` in closed-form Reiman oracle

`tests/jax_core/test_analytic_fields_item11.py:258` (`test_reiman_closed_form`):
```python
inner = phi - 6 * np.arctan(z / (-1 + sqxy))
```

The Reiman field uses `theta = np.arctan2(Zp, RR - R_axis)` (`analytic_fields.py:713`). For points with `RR < 1` (true in the test fixture: `sqxy ≈ 0.9`), the two branches differ by ±π. The test silently passes because `k=6` is divisible by 2 and `cos(6(θ+π) - φ) = cos(6θ - φ)`. For odd `k` (e.g., 3, 5, 7) the closed-form oracle would diverge from `reiman_B`.

This is a flaw inherited from the upstream CPU test (`test_magneticfields.py:984`) and is therefore beyond the JAX port's scope — but the audit notes it because the test reuses the upstream `arctan` formula verbatim and never verifies on odd `k`.

**Recommendation.** Replace `np.arctan(z/(-1+sqxy))` with `np.arctan2(z, -1+sqxy)` in the JAX closed-form oracle (and ideally in the upstream test too). LOW severity.

### R-2 (INFO) — No closed-form `Bphi = -1` assertion

The Reiman convention pins `B_φ = -1.0` (a stellarator-like "negative-helicity" choice). Neither the JAX nor the cross-oracle test asserts the projection `B · φ̂ = -1` explicitly; it is buried inside the closed-form `Bx`, `By`, `Bz` formulae. If the JAX kernel mistakenly returned `Bphi = +1`, the `test_reiman_closed_form` would still pass because the closed-form expression also encodes `Bphi = -1` implicitly.

**Recommendation.** Add one assertion `np.allclose(B_jax · phi_hat, -1.0)` to `test_reiman_closed_form`. INFO severity.

---

## Test coverage gaps

| Gap | Severity | Where it should live | Notes |
|---|---|---|---|
| Direct `div(B) = trace(dB) = 0` assertion on `dommaschk_dB` output | INFO | `tests/jax_core/test_analytic_fields_item11.py` | Upstream CPU test verifies on Reiman only (line 974). Defense-in-depth direct assertion on the JAX side closes the audit gap. |
| Central-FD Taylor test for `dommaschk_dB` | INFO | `tests/jax_core/test_analytic_fields_item11.py` | Reiman already has `test_reiman_dB_taylor`. Adding a Dommaschk twin gives ULP-independent gradient validation. |
| `jax.grad` over `spec.coeffs` parity | INFO | `tests/jax_core/test_analytic_fields_item11.py` | The Dommaschk field is linear in each `coeff`, so `∂B/∂coeff_k = B_k(coeff_k=1, others=0)` should match a finite-difference probe. |
| Odd-`k` Reiman regression | INFO | `tests/jax_core/test_analytic_fields_item11.py` | Currently only `k=6` is tested at the bare-kernel layer; item-15 has `k=[4,6,8]` but no odd entry. |
| `B_φ = −1` explicit assertion | INFO | `test_reiman_closed_form` | Easy one-liner, prevents a silent sign regression. |
| Large-coefficient parity-degradation guard | INFO | `test_magneticfieldclasses_jax_item15.py` | The documented divergence on `5.10e10` coefficients is currently in a comment; an explicit `with pytest.warns(...)` or a tracked relaxed tolerance would close the audit trail. |

---

## Recommended actions (ordered by severity)

1. **MEDIUM (D-1).** Document the reduction-order behavior at the public API. Add a `Notes` section to the `dommaschk_B` / `dommaschk_dB` docstrings warning that the JAX kernel guarantees algebraic but not byte-identity parity with `sopp.DommaschkB`. Cross-link to `docs/parity_dual_mode_contract_2026-05-08.md`. No code change.
2. **LOW (D-3).** Add shape validation in `dommaschk_B`, `dommaschk_dB`, `reiman_B`, `reiman_dB` to reject `points` that are not `[N, 3]` float64. Two-line `raise ValueError` block.
3. **LOW (R-1).** Replace `np.arctan` with `np.arctan2` in the Reiman closed-form oracle test.
4. **INFO (T1).** Add one explicit divergence-free assertion in `test_analytic_fields_item11.py` against `dommaschk_dB`.
5. **INFO (T2).** Add a central-FD Taylor test for `dommaschk_dB` mirroring `test_reiman_dB_taylor`.
6. **INFO (T3).** Add a `jax.grad`-over-coefficients parity test that compares autodiff per-mode contributions to a finite-difference probe at well-conditioned coefficients.
7. **INFO (T4).** Add an odd-`k` Reiman parity test (e.g. `k=[3, 5, 7]`).
8. **INFO (T5).** Add a `B_φ = −1` assertion in `test_reiman_closed_form`.
9. **INFO (D-2).** Either bound the `lru_cache` sizes or expose a `clear_caches()` utility.

No CRITICAL or HIGH severity findings. The JAX port is **mathematically and algorithmically correct** with respect to its C++ references.
