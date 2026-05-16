# Priority 7 — PM optimization DEEPER (second-pass) audit

**Audit timestamp**: 2026-05-16
**Auditor**: Claude (fresh second-pass agent)
**Scope**: hunt for second-order issues that the forward-parity first pass
would have systematically missed. The first-pass artifact
`07_pm_optimization.md` already verified forward formulas and identified
the H7 `projection_l2_balls` NaN footgun and the documented MwPGP
early-exit/history omissions.

## Files audited

| Path | Lines | Role |
|------|-------|------|
| `src/simsopt/jax_core/pm_optimization.py` | 2485 | JAX kernel |
| `src/simsoptpp/permanent_magnet_optimization.cpp` | 1332 | C++ reference |
| `src/simsoptpp/permanent_magnet_optimization.h` | 39 | C++ header |
| `src/simsopt/solve/permanent_magnet_optimization.py` | 480 | CPU orchestrator |
| `src/simsopt/solve/permanent_magnet_optimization_jax.py` | 776 | JAX orchestrator |
| `tests/jax_core/test_pm_optimization_jax_item25.py` | 1495 | JAX kernel parity tests |
| `tests/solve/test_permanent_magnet_optimization_jax_item28.py` | 878 | JAX orchestrator tests |

## Executive summary — top deeper findings

| # | Severity | Finding | Citation |
|---|----------|---------|----------|
| D1 | **HIGH (corroborate + extend H7)** | The H7 fix in the prompt corrigendum is mandatory: `unit = jnp.ones_like(m_maxima)` ALONE is insufficient. With `m = zeros` AND `m_maxima_i = 0` (which is exactly the initial state used by `mwpgp_initial_state` when the orchestrator defaults `m0 = zeros`), `norm/m_maxima = 0/0 = NaN`, and `jnp.maximum(1.0, NaN) = NaN`. The full fix must also switch to `jnp.fmax`. The same NaN bug is also latent in the CPU NumPy orchestrator (`projection_L2_balls`, `solve/permanent_magnet_optimization.py:83` uses `np.maximum(np.ones(...), denom_fac)` which propagates NaN). | `pm_optimization.py:2122-2125`; `solve/permanent_magnet_optimization.py:80-84`; reproduced live (see §D1 below) |
| D2 | **HIGH** | **`1e50` sentinel collision is REAL and reproducible.** With `||b|| ~ 1e26` (not implausible for poorly-scaled real PM problems), real available-slot GPMO candidate costs reach `~1.27e52`, which is **above** the `1e50` "unavailable" sentinel. `jnp.argmin` then returns the FIRST unavailable-sentinel slot and the solver silently `set`-writes into an already-placed dipole. No test scales `b` past `O(1)` to detect this. Affects all 5 GPMO variants (baseline, multi, backtracking, ArbVec, ArbVec_backtracking). | `pm_optimization.py:609, 776, 1063, 1598`; reproduced (see §D2) |
| D3 | **HIGH** | **`nu = 0` produces silent NaN** through `1.0/(2.0 * nu) = +inf` in `_hessian_action` (line 2271) AND through `ATb_rs = ATb + m_proxy / nu` (line 2464). Result: every entry of `m_final` is `NaN` after a single MwPGP step. **No validator** rejects `nu <= 0` (cf. `_validate_gpmo_static_args`, `_validate_gpmo_backtracking_static_args`). The CPU orchestrator at `solve/permanent_magnet_optimization.py:118-275` similarly does not validate `nu`. | `pm_optimization.py:2271, 2407, 2464`; reproduced (see §D3) |
| D4 | **MEDIUM** | **`nu < 0` is silently accepted**, runs to convergence-ish, and produces a non-convex optimization. The Hessian action becomes `H = A^T A + 2 reg_l2 I - 1/|nu| I`; if `2 reg_l2 + 1/(2nu) < 0` the objective is unbounded below on the L2 ball. No guard. | `pm_optimization.py:2271`; reproduced (see §D4) |
| D5 | **MEDIUM** | **Argmin treats NaN as smallest.** If any kernel evaluation introduces NaN (e.g., via D1's projection NaN, the D3 nu hazard, or a future maintenance regression that lets NaN leak into `costs`), every GPMO variant will silently *prefer* the NaN-poisoned dipole. There is no `jnp.where(jnp.isfinite(costs), costs, sentinel)` guard before `jnp.argmin`. | `pm_optimization.py:625, 792, 1267, 1617, 1915`; reproduced (see §D5) |
| D6 | **MEDIUM** | **MwPGP with `alpha > 2/lambda_max(H)` oscillates wildly instead of diverging.** The L2-ball projection acts as a clamp, so the residual history shows decreasing-then-blowing-up-then-bounded oscillation (`-5 → -8 → +7 → +29 → +101 → ...`). No guard, no diagnostic, and the orchestrator passes a fixed `alpha = 2/ATA_scale * (1 - 1e-5)` from `solve/permanent_magnet_optimization.py:197` that depends solely on `pm_opt.ATA_scale`. If the caller substitutes a custom step size or modifies `ATA_scale` post hoc, divergence is silent. | `pm_optimization.py:2278-2330`; reproduced (see §D6) |
| D7 | **MEDIUM** | **Every change in `N` (number of dipoles) or `P` (polarization vector count) triggers a full JAX recompile (≈100 ms each).** For autoresearch sweeps that vary grid resolution this is significant; for the FAMUS workflow that materializes a single `PermanentMagnetGrid` it is not. The kernel is NOT shape-polymorphic and uses `int(m_maxima.shape[0])` (line 686, 854, 1387, 1690, 2014) to force Python-level int extraction. | `pm_optimization.py:686, 854, 1387, 1690, 2014`; reproduced (see §D7) |
| D8 | **LOW** | **GPMO_backtracking with `single_direction ∈ {0,1,2}` is uncovered** but works correctly. Verified manually (see §D8). The first pass flagged this gap. Adding tests would convert "untested-but-correct" to "asserted-correct" before someone touches `_single_direction_mask` again. | `pm_optimization.py:1908-1913`; verified manually |
| D9 | **LOW** | **`expand_branch` is reachable** with a fixture where `alpha_cg ≥ alpha_f`. First pass noted "static jaxpr-count of `cond` does not prove execution" — verified that a synthetic `m=[0.95,0,0], m_maxima=1, ATb=[-10,0,0], alpha=1e-4` drives the iterate to `[-1,0,0]` (projected onto L2 ball boundary), confirming expand_branch execution. | `pm_optimization.py:2317-2321`; reproduced (see §D9) |
| D10 | **INFO** | **C++ `print_*` reentrance**: stdout prints in the C++ kernel are not mirrored by JAX. No existing test captures stdout. If any downstream caller parses the C++ printed line `"%d wyrms removed out of %d possible dipoles\n"` (cpp:526) or `"MwPGP algorithm ended early, at iteration %d\n"` (cpp:320), the JAX path will silently produce no output. Not a parity bug, but a behavior-mismatch surface. | `cpp:144, 320, 349, 526, 939` |
| D11 | **INFO** | **GPMO_multi/ArbVec penalty-index quirk has no buffer-overrun in either kernel.** C++ `mmax_ptr[cj]` reads element `cj ∈ [0, N)` of the `3N`-long component-expanded vector (`np.sqrt(reg_l2) * np.repeat(m_maxima, 3)`). No overrun since `cj < N ≤ 3N`. JAX mirrors via `_component_mmax(m_maxima)[:m_maxima.shape[0]]` (line 771, 1057) which also reads the first N entries. Both kernels report the L2 penalty as `reg_l2 * m_maxima[cj]^2` — which is the user-facing behavior since `repeat(m_maxima, 3)[0::3] == repeat(m_maxima, 3)[1::3] == repeat(m_maxima, 3)[2::3] == m_maxima`. | `cpp:662, 822, 1193`; `pm_optimization.py:771, 1057, 1591` |
| D12 | **INFO** | **`backtracking = 0` is properly rejected** for both `GPMO_backtracking` (`pm_optimization.py:498-499`) and `GPMO_ArbVec_backtracking` (`pm_optimization.py:556-557`). No `k % 0` hazard in the dewyrming gate at lines 1931, 1292. The C++ has no such guard but `k >= backtracking` (cpp:487) would silently bypass dewyrming when `backtracking <= 0`. | `pm_optimization.py:498, 556` |
| D13 | **INFO** | **GPU memory for large N (10^5)**: `A` itself is `M × 3N` (e.g., 1000 × 300000 ≈ 2.4 GB float64). State arrays are negligible (`N × 3 ≈ 2.4 MB`). The closed-form Hessian action (`pm_optimization.py:2256-2272`) never materializes `A^T A` (would be 720 GB), so the in-loop memory is dominated by `A`. **`jacfwd` / `grad` over `mwpgp_solve` are not currently used in the M5 single-stage adapter pipeline**, but if a future caller wraps `mwpgp_solve` in an autodiff context, `lax.scan` would force checkpointing on the (n_steps, N, 3) state trajectory and could blow up CPU/GPU memory by `n_steps` × `(N, 3)` ≈ a few GB for `n_steps=1000`. No such usage exists today. | `pm_optimization.py:2256-2272, 2477` |
| D14 | **INFO** | **`safe_norm` in `_beta_tilde` is bullet-proof for the intended path but the `_on_ball(zeros, m_maxima=0)` case still bypasses the guard.** When `m=zeros` AND `m_maxima_i=0`, `_on_ball` returns `True` (|0-0| < 1e-8 + 0 = True), so `_beta_tilde` enters the on-ball branch with `safe_norm = 1.0`. Then `ng = mg / 1 = 0`, so the branch is `ng > 0 ? g : grg = g_reduced_gradient(...)`. The `g_reduced_gradient` call funnels into `projection_l2_balls` which produces NaN per D1. So `_beta_tilde` itself is fine; the NaN re-enters via `g_reduced_gradient → projection_l2_balls`. | `pm_optimization.py:2128-2192`; reproduced (see §D14) |

The H7 NaN bug, the sentinel collision, and the `nu` validator gaps are
the items most likely to bite a real user. D1+D2+D3+D5 share a common
"silently produces NaN or wrong placement" failure mode and warrant a
unified hardening pass.

## §D1 — projection_l2_balls NaN: corrigendum verification

### Live reproduction (jax 0.10.0, jaxlib 0.10.0)

```
m = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
m_maxima = [1.0, 0.0, 1.0]

# Current (buggy) line 2123:
unit_buggy = m_maxima / m_maxima      = [ 1.  nan  1.]
denom_buggy = jnp.maximum(unit_buggy, norm/m_maxima)
                                      = [3.7416575  nan  1.0]
out_buggy = m / denom_buggy[:, None]
                                      = [[0.267, 0.535, 0.802],
                                         [ nan,  nan,  nan],
                                         [0.5,  0.5,  0.5]]
```

The corrigendum claims `unit = jnp.ones_like(m_maxima)` alone is
insufficient. Verified:

```
# Partial-fix (ones_like) but still jnp.maximum:
unit_correct = [1., 1.]      # for the m=zeros row at m_maxima=0
norm/m_maxima = [3.7..., 0/0 = NaN]  # if m=zeros AND m_maxima=0 → 0/0 = NaN
jnp.maximum(1.0, NaN) = NaN
```

So `m_maxima_i = 0` with `m_i = zeros` (the orchestrator default state
in `mwpgp_initial_state` when `m0 = zeros`) still produces NaN even
with `unit = jnp.ones_like(m_maxima)`. The full fix is:

```python
def projection_l2_balls(m, m_maxima):
    norm = jnp.linalg.norm(m, axis=1)
    unit = jnp.ones_like(m_maxima)
    safe_m_maxima = jnp.where(m_maxima > 0.0, m_maxima, 1.0)
    ratio = norm / safe_m_maxima
    denom = jnp.fmax(unit, ratio)  # fmax ignores NaN
    return m / denom[:, None]
```

`jnp.fmax` returns the non-NaN value when one arg is NaN; combined with
the `safe_m_maxima` guard (which keeps the division finite) this is
correct for all cases including `m_maxima_i = 0`.

### CPU orchestrator has the same hazard

`solve/permanent_magnet_optimization.py:80-84`:

```python
def projection_L2_balls(x, mmax):
    N = len(x) // 3
    x_shaped = x.reshape(N, 3)
    denom_fac = np.sqrt(np.sum(x_shaped ** 2, axis=-1)) / mmax
    denom = np.maximum(np.ones(len(denom_fac)), denom_fac)
    return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)
```

`np.maximum(1.0, NaN) → NaN`. The CPU and JAX both produce NaN on the
same input. **The C++ kernel `projection_L2_balls` (cpp:13) survives**
because `std::max(1.0, NaN)` is implementation-defined and typically
returns the first argument (`1.0`) under GCC. So the C++ ↔ Python
parity is already broken on this edge case, but the test suite never
exercises it.

### Untested edge cases (D1)

1. **`m_maxima = [1.0, 0.0, 1.0]`, `m = random`** — NaN in row 1 propagates through `mwpgp_solve` to every output. Add test asserting `jnp.all(jnp.isfinite(out))` for this input.
2. **`m_maxima = [0.0, 0.0, 0.0]`, `m = zeros`** — JAX kernel returns all-NaN; C++ returns all-zero. Add test or document divergence.
3. **`reg_l2 = 0`, `m_maxima = 0` in GPMO**: `_component_mmax(m_maxima) = zeros`, `penalty = zeros`. No division by zero, but argmin operates on the unmodified residual cost. Test that GPMO doesn't silently misbehave when `m_maxima = 0` for some dipoles.

## §D2 — 1e50 sentinel collision: reproducible bug

### Live reproduction (M=5, N=3, ||b|| ~ 1e26)

```
A_scaled = N(0,1) random (5, 9)
b ~ N(0,1)*1e26 (so ||b||^2 ~ 1e52)
available = [[False, False, False], [True, True, True], [True, True, True]]
              ^^^^^ dipole 0 unavailable (already placed)

gpmo_baseline_candidate_costs output (length 18):
  costs[0:3]  = [1.0e50, 1.0e50, 1.0e50]  # dipole 0 sentinel (unavailable)
  costs[3:9]  = [1.10e53, ...]              # dipoles 1,2 real costs (>> 1e50)
  costs[9:12] = [1.0e50, 1.0e50, 1.0e50]  # dipole 0 sentinel (minus)
  costs[12:18]= [1.10e53, ...]              # dipoles 1,2 real costs (minus)

jnp.argmin(costs) = 0
Decoded: dipole=0, component=0, sign=+1.0 → UNAVAILABLE DIPOLE
```

The solver writes `x[0, 0] = +1.0` on top of whatever was already there, marks dipole 0 unavailable again (already was), and produces a residual update that double-counts dipole 0's contribution.

### How does this happen in practice?

Real-world `b` is sometimes scaled by `Bn_target / (mu_0 / 4π)`. For
HBT-class magnets with `||B_target||^2 ~ 1 T^2` and integration over a
plasma boundary at ~1 m radius, `||b||^2 = O(1)`. So **the test fixture
range (`||b|| ~ 1`) is realistic and the sentinel hazard does not
manifest in production CI**. However:

1. **Numerical experimentation with `b` scaled to other unit systems** (e.g., raw flux without mu_0 rescaling) could push `||b||` past 10^25.
2. **Outer relax-and-split iterations** can grow `m_proxy` to large values, but `m_proxy` enters MwPGP (not GPMO), so this hazard is specific to GPMO.
3. **Code refactoring** (e.g., a maintenance edit that scales `A_scaled` differently) could silently introduce the hazard.

### Recommended hardening

Replace the literal `1.0e50` sentinel with `+jnp.inf`:

```python
sentinel = jnp.asarray(jnp.inf, dtype=A_arr.dtype)
plus = jnp.where(allowed, plus, sentinel)
minus = jnp.where(allowed, minus, sentinel)
```

`jnp.argmin` on a vector that includes `+inf` returns a finite-cost
index correctly. The only caveat: if **all** candidates are `inf`
(everything unavailable), `argmin` returns 0 deterministically — same
behavior as `1e50` but bulletproof to magnitude scaling.

### Untested edge cases (D2)

1. **Scaling `A_scaled` and `b` by 1e13** — real costs reach 1e26+, but still within float64 range. Currently safe with `1e50`. Test asserts `jnp.argmin(costs) == expected` for this scaling.
2. **Scaling by 1e25** — real costs reach 1e50+. Currently produces wrong placement; would be fixed by inf-sentinel.
3. **Pathological `reg_l2 = 1e60`** — penalty alone exceeds `1e50`. Currently produces wrong placement; same fix.

## §D3 — nu = 0 silent NaN

### Live reproduction (n_steps=3, nu=0)

```
spec.nu = 0
ATb_rs = ATb + m_proxy / 0  →  inf (or NaN if m_proxy = 0)
H v action: scale = 2 * (reg_l2 + 1/(2*0)) = +inf
Hv = AtAv + inf*v_flat = inf (or NaN at zero entries)

mwpgp_solve output (n_steps=3):
  m_final = [[nan, nan, nan], [nan, nan, nan], ...]
```

Empirically: with `nu = 1e-12`, the output is tiny but finite (m ~
1e-12). The transition is sharp at `nu = 0`.

### Recommendation

Add a Python-level validator:

```python
def _validate_pm_spec(spec: PMOptimizationSpec) -> None:
    if not (float(spec.nu) > 0):
        raise ValueError(f"nu must be strictly positive; got {float(spec.nu)}")
```

Note: cannot validate inside a JIT (would force a host conversion).
Validate at the `mwpgp_solve` entry, before `_run_mwpgp` is called.

### Untested edge cases (D3)

1. `nu = 0`: not tested anywhere; produces silent NaN.
2. `nu = +0` vs `nu = -0`: IEEE-754 distinguishes these and `1/(-0)` = `-inf`. Probably equivalent in outcome but undocumented.
3. `nu = inf`: tested via the default `1e100`, but `nu = jnp.inf` literal is untested. `1/(2*inf) = 0`, so safe.
4. `nu = NaN`: untested; would propagate.

## §D4 — nu < 0 silent acceptance

### Live reproduction (n_steps=3, nu=-1)

```
spec.nu = -1.0
1.0/(2.0 * -1.0) = -0.5  →  H = A^T A - 1.0 * I (subtracts the L2 mass)

Output: finite (no NaN), e.g. m_final = [[-0.157, -0.018, +0.293], ...]
But the optimization is now non-convex (Hessian has negative
eigenvalues if A^T A has eigenvalues < 1).
```

No validator, no test. The convex-step machinery silently solves a
non-convex problem and may converge to a saddle.

### Recommendation

Same as D3: gate `spec.nu > 0`.

## §D5 — argmin treats NaN as smallest

### Live reproduction

```
jnp.argmin([2.0, NaN, 1.0]) = 1     # NaN preferred over the real min
```

This is documented JAX/numpy behavior: argmin propagates NaN selection.
Combined with D1 (projection NaN), D3 (nu=0 NaN), or any future leak,
this means **any NaN-poisoned dipole would be silently selected** by
all 5 GPMO variants. The kernel has no `jnp.where(jnp.isfinite(costs),
costs, sentinel)` guard before argmin.

### Recommendation

Add a defensive guard after candidate cost computation:

```python
costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)
choice = jnp.argmin(costs)
```

This is a single-line defense and has zero overhead (single `where`).

## §D6 — MwPGP oscillation with too-large alpha

### Live reproduction (alpha=100, alpha_safe=0.025)

```
residual_history = [-5.4, -7.2, -8.2, -8.4, -8.6, -8.9, +7.2,
                   +30.0, +101.6, +54.3, +130.3, +84.8, +152.9,
                   +113.8, +151.1, +121.5, +148.9, +123.3, +148.0, +123.8]
m_final = finite, bounded by L2 ball (norms ≤ m_maxima).
```

The L2-ball projection prevents iterate divergence to infinity, but the
**objective oscillates rather than decreases**. The
`test_cost_monotone_decreasing` test (item 25 line 1116) uses
`alpha = 1/sigma_max(A)^2` (very conservative) and would pass even if
the kernel had an `alpha`-clamping bug, because the safe step never
triggers oscillation.

### Recommendation

Add a runtime invariant assertion (only via `jax.debug.check` since
this is a tracer-level invariant):

```python
# At construction time (Python level):
if float(spec.alpha) <= 0:
    raise ValueError("alpha must be positive")
# At runtime — optional diagnostic:
# (Skip; runtime checks slow JIT)
```

The simpler fix: document that the caller MUST pass `alpha < 2 /
lambda_max(H)` and add a test that asserts oscillation for `alpha = 10
× alpha_safe` (to catch a future regression that silently changes the
algorithm).

## §D7 — Recompilation cost for varying N or P

### Live reproduction

```
@jax.jit
def run(spec, A, ATb, m0):
    return mwpgp_solve(spec, A, ATb, m0, n_steps=2)[0]

run(N=3, M=9)  → 0.163 s compile, 0.000 s steady-state
run(N=3, M=9) again → 0.000 s (cached)
run(N=5, M=15) → 0.140 s recompile (different shape)
run(N=10, M=30) → 0.135 s recompile
```

The kernel uses `int(m_maxima.shape[0])` at lines 686, 854, 1387, 1690,
2014 to force Python-side int extraction. This makes shape-polymorphism
impossible. For autoresearch sweeps over `N`, each sweep step pays a
~150 ms compile penalty.

### Recommendation

Two options:

1. **Accept the cost** — single-stage FAMUS-style PM problems fix `N` once. Most production callers compile once and run many outer iterations.
2. **Refactor to shape-polymorphism** — replace `int(m_maxima.shape[0])` with `m_maxima.shape[0]` (tracer-friendly) and use `jax.numpy.reshape(m_maxima, (-1,))`. Would require auditing `_validate_*_static_args` to allow shapes to be inferred at trace time. Significant churn; only worthwhile if a sweep over `N` is a common workflow.

The compile cost is *not* a correctness issue, just a UX/throughput
issue.

## §D8 — GPMO_backtracking with single_direction ∈ {0,1,2}

### Manual verification

```python
for single_direction in (-1, 0, 1, 2):
    x_cpp = simsoptpp.GPMO_backtracking(..., single_direction=single_direction)
    result = gpmo_backtracking_solve(spec_with_single_direction, ...)
    assert max(|result.x - x_cpp|) == 0.0
```

Verified: all 4 values yield byte-identical state-trace parity with
C++. The first pass correctly flagged this as INFO (untested but
working). Mirror the existing
`test_solver_matches_cpp_baseline_for_all_single_direction_modes`
pattern at line 331-359 to lock down the contract.

## §D9 — expand_branch is reachable

### Manual construction

```
N=1, A=I_3, m0=[0.95, 0, 0], m_maxima=1, ATb=[-10, 0, 0], alpha=1e-4

Then:
g_init = A^T A m0 + 2*(reg_l2 + 1/(2nu))*m0 - ATb
       = m0 - ATb
       = [10.95, 0, 0]
p_init = phi(m0, g_init) with m0 off-ball (|0.9025 - 1| = 0.0975 > 1e-8 + 1e-5)
       = g_init
       = [10.95, 0, 0]

Step:
g_alpha_p = g_reduced_projected_gradient(...)
phi_g = phi(...)
norm_g_alpha_p == norm_phi (equal because off-ball, beta=0)
inner = True

inner_branch:
  ATAp = H p = p = [10.95, 0, 0]
  gp = 10.95^2
  pATAp = 10.95^2
  alpha_cg = 1.0
  alpha_f = solution of a t^2 + b t + c = 0
          a = ||p||^2 = 119.9, b = -2 m·p = -20.805, c = ||m||^2 - 1 = -0.0975
          alpha_f = (20.805 + sqrt(432.84 + 46.76)) / 239.8 ≈ 0.178
  alpha_cg (=1.0) >= alpha_f (=0.178) → expand_branch
```

Verified output: `m_step = [-1.0, 0.0, 0.0]` (projected to L2 boundary).

Recommended test addition (drop-in for item 25):

```python
def test_step_body_takes_expand_branch_when_cg_overshoots_ball(self):
    A = np.eye(3)
    ATb = np.array([[-10.0, 0.0, 0.0]])
    m_maxima = np.array([1.0])
    m_proxy = np.zeros((1, 3))
    m0 = np.array([[0.95, 0.0, 0.0]])
    spec = _make_spec(m_maxima, m_proxy, alpha=1e-4, reg_l2=0.0, nu=1.0e100)
    state = mwpgp_initial_state(spec, jnp.asarray(A), jnp.asarray(ATb), jnp.asarray(m0))
    new_state = mwpgp_step(spec, state, jnp.asarray(A), jnp.asarray(ATb))
    # Expand branch projects onto the L2 boundary at the opposite side.
    np.testing.assert_allclose(np.asarray(new_state[0])[0], [-1.0, 0.0, 0.0])
```

## §D10 — C++ print reentrance

C++ prints diagnostics at:

- `cpp:144` (MwPGP per-iteration breakdown): `"%d ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e \n"`
- `cpp:320` (MwPGP early exit): `"MwPGP algorithm ended early, at iteration %d\n"`
- `cpp:349` (GPMO printing): `"%d ... %.2e ... %.2e \n"`
- `cpp:526, 939` (dewyrming counts): `"%d wyrms removed out of %d possible dipoles\n"` / `"Backtracking: %d wyrms removed\n"`
- `cpp:543, 567-573, 943-979` (GPMO termination diagnostics)
- `cpp:1112-1116` (initialize_GPMO_ArbVec out-of-tolerance warning)

The JAX kernel produces NO stdout. No test captures stdout. Two
risks:

1. Downstream scripts that parse the C++ output (e.g., `grep` for "wyrms removed") would break under the JAX path.
2. The `print_*` `print_iter` counter and the `m_history`/`objective_history` arrays in C++ are tied together: each print advances `print_iter` and updates the history slot at that index. The JAX path returns the full per-step residual proxy under `lax.scan`, not the sparse 21-slot snapshot. A caller mixing C++ history shape expectations with JAX output will index past the end of the JAX history.

### Recommendation

Document in the `mwpgp_solve` and `gpmo_*_solve` docstrings: "Does not
produce stdout. To compare against C++ verbose output, set `verbose=False`
on the C++ side."

## §D11 — Penalty-index quirk: NO buffer overrun

### C++ pattern

```c++
// GPMO_multi, cpp:662
mmax_partial_sum += mmax_ptr[cj] * mmax_ptr[cj];   // cj is a DIPOLE id ∈ [0, N)
// mmax is a 3N-long vector: np.sqrt(reg_l2) * np.repeat(m_maxima, 3)
```

Reading element `cj` (range `[0, N)`) of a vector with length `3N` is
safe: `cj < N ≤ 3N`. The "quirk" is semantic — the C++ author probably
intended `mmax_ptr[3 * cj + l]` or similar, but since `repeat(m_maxima,
3)` triples each entry, `mmax_ptr[cj]` happens to equal
`m_maxima[cj/3]` *for `cj < 3*floor(N/3)`* but is actually
`m_maxima[cj]` because of the triple-repeat. So for `cj ∈ [0, N)`,
`mmax_ptr[cj] = m_maxima[cj]`. No bug; just a misnamed pointer.

### JAX mirror

```python
# pm_optimization.py:771
penalty = reg_l2 * _component_mmax(m_maxima)[: m_maxima.shape[0]] ** 2
# Equivalent to: reg_l2 * m_maxima ** 2
```

Equivalent semantics. The first-pass audit was correct: this is faithful.

### Edge-case test recommendation

Add a regression test with **non-uniform** `m_maxima` (e.g., `[0.3,
0.7, 1.0, 0.5]`) and verify that the JAX kernel matches C++. A future
"fix" that switches to `mmax_ptr[3*cj + l]` would silently change the
penalty per dipole and the test would catch it.

## §D12 — Backtracking = 0 guard

Verified at JAX level:

```
GPMOBacktrackingSpec(backtracking=0) → raises ValueError("backtracking must be positive; got 0")
GPMOArbVecBacktrackingSpec(backtracking=0) → raises ValueError(...)
```

`pm_optimization.py:498-499`:
```python
if backtracking < 1:
    raise ValueError(f"backtracking must be positive; got {backtracking}")
```

So `k % 0` is unreachable in JAX. **C++ has no such guard** — passing
`backtracking=0` to `simsoptpp.GPMO_backtracking` would silently
bypass the dewyrming gate (`k >= 0 && k % 0`) — `k % 0` is undefined
behavior in C++. Not a parity bug (the JAX guard is stricter), but
worth noting that the C++ kernel is fragile here.

## §D13 — Large-N memory analysis

For `N = 10^5, M = 10^3`:

| Array | Shape | Bytes (float64) |
|-------|-------|-----------------|
| `A` | (1000, 300000) | 2.4 GB |
| `A^T A` (never materialized) | (300000, 300000) | 720 GB |
| `m`, `g`, `p`, `ATAp` (state) | (100000, 3) each | 2.4 MB each |
| `residual_history` (scan output) | (n_steps,) | n_steps × 8 B |

The Hessian action `_hessian_action` (line 2256) computes `A @ v_flat`
then `A.T @ Av`, materializing only the `(M,)` intermediate Av and the
`(3N,)` AtAv. No memory blow-up.

**No `jacfwd`/`grad` over `mwpgp_solve` currently**. The M5 single-stage
adapter (`surfaceobjectives_jax.py`) does not call into `mwpgp_solve`;
the PM optimizer is invoked via `relax_and_split_jax` in a forward-only
pipeline. So the 21-slot history dropping is not a concern.

If a future caller wraps `mwpgp_solve` in `jacfwd`, the scan-vmap rule
would materialize one tangent per scan step: `(n_steps × N × 3 × 8)` =
`n_steps × 2.4 MB`. For `n_steps = 1000`, that's 2.4 GB tangent
memory, manageable on a 24 GB GPU but not on a 16 GB laptop.

## §D14 — beta_tilde safe_norm: bullet-proof?

The first pass said "safe_norm guards ||m||=0 in beta_tilde justified by
on_ball mask". Verification:

```python
def _beta_tilde(m, g, alpha, m_maxima):
    on_ball = _on_ball(m, m_maxima)  # |||m||^2 - m_maxima^2| < 1e-8 + 1e-5 m_maxima^2
    norm = jnp.linalg.norm(m, axis=1)
    safe_norm = jnp.where(norm > 0.0, norm, 1.0)
    mg = jnp.sum(m * g, axis=1)
    ng = mg / safe_norm
    grg = g_reduced_gradient(m, g, alpha, m_maxima)  # ← funnel to projection_l2_balls
    on_active_grad = jnp.where((ng > 0.0)[:, None], g, grg)
    return jnp.where(on_ball[:, None], on_active_grad, 0.0)
```

Two cases:

1. **Intended use**: `m_maxima > 0`, `||m|| > 0`, on_ball=True only when `||m|| ≈ m_maxima > 0`. `safe_norm = ||m|| > 0`, `ng = finite`. **Bullet-proof.**
2. **D1 case**: `m_maxima = 0`, `m = zeros`. Then:
   - `||m||^2 - m_maxima^2 = 0 - 0 = 0`
   - `1e-8 + 1e-5 * 0 = 1e-8`
   - `|0| < 1e-8` → `on_ball = True`. Wrong: a zero-radius ball at zero is a degenerate point, but the predicate fires.
   - `safe_norm = where(0 > 0, 0, 1) = 1`, `ng = 0/1 = 0`, `ng > 0` false.
   - Falls through to `grg = g_reduced_gradient(zeros, g, alpha, zeros)` → `projection_l2_balls(zeros - alpha*g, zeros)` → NaN (per D1).

So **`safe_norm` itself is bullet-proof**, but the same `m_maxima = 0`
input triggers NaN through the `grg` codepath. The fix is upstream
(D1 hardening of `projection_l2_balls`); `_beta_tilde` need not change.

Verified live:

```
m=zeros, m_maxima=0:
g_reduced_projected_gradient = [[nan, nan, nan]]
```

## Test coverage gap inventory (consolidated)

| Gap | Severity | Suggested test |
|-----|----------|----------------|
| `projection_l2_balls` with any `m_maxima_i = 0` | HIGH | `out = projection_l2_balls(jnp.zeros((3,3)), jnp.array([1.0, 0.0, 1.0])); assert jnp.all(jnp.isfinite(out))` |
| GPMO with `||b||^2 > 1e50` | HIGH | Build a fixture with `b = 1e26 * random`, assert `argmin` picks an available dipole |
| MwPGP with `nu = 0`, `nu < 0`, `nu = NaN` | HIGH | Add validator + assert ValueError on these inputs |
| MwPGP with `alpha = 0`, `alpha < 0` | MEDIUM | Add validator (alpha > 0); current code accepts negative alpha |
| MwPGP with `alpha = 100 × alpha_safe` (oscillation) | MEDIUM | Assert cost monotonicity fails or warning emitted |
| GPMO_backtracking with `single_direction ∈ {0,1,2}` | LOW | Mirror baseline pattern |
| `expand_branch` execution coverage | LOW | The §D9 construction is a drop-in test |
| Non-uniform `m_maxima` for GPMO_multi/ArbVec penalty | LOW | Assert match against C++ with `m_maxima = [0.3, 0.7, 1.0, 0.5, 0.4]` |
| End-to-end `relax_and_split_jax` vs CPU with `reg_l0 > 0` over ≥2 outer iters | INFO | Currently only spot-checked at the prox primitives |
| GPMO with non-finite costs (synthetic NaN injection) | INFO | Verify argmin guard once added |
| `m_init` with infeasible dipoles for `setup_initial_condition_jax` | INFO | Currently rejects via `_raise_if_infeasible_initial_condition`; test does not cover the disallowed-transfer-guard path |

## Recommendations (priority order)

1. **Fix `projection_l2_balls`** to use `jnp.fmax` AND `jnp.ones_like` (or equivalently a safe-division guard on `m_maxima`). Add regression test for `m_maxima_i = 0` cases. CPU orchestrator `projection_L2_balls` should also be hardened.
2. **Add validators** for `nu > 0` and `alpha > 0` at the public entry point `mwpgp_solve` and `relax_and_split_jax`. The validation cannot be inside a JIT region, so it must happen at the Python-callable boundary.
3. **Switch `1e50` sentinel to `jnp.inf`** in all 5 GPMO candidate-cost functions (lines 609, 776, 1063, 1598; plus the `R2s` re-assignments at lines 614, 779, 1066). Both `argmin` and `where` handle inf correctly. Add a regression test scaling `||b||` past `1e25`.
4. **Add a defensive `jnp.isfinite` guard** before every `jnp.argmin` on GPMO candidate costs (lines 625, 792, 1267, 1617, 1915). Single-line `costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)`. Zero overhead.
5. **Add `single_direction` sweep** for `GPMO_backtracking` mirroring the existing baseline pattern.
6. **Add `expand_branch` execution test** using the §D9 fixture.
7. **Add non-uniform `m_maxima` GPMO_multi/ArbVec regression test** to lock down the penalty-index quirk.
8. **Document the recompilation cost** for varying `N`/`P` in the kernel docstring so users know about the 100 ms overhead per shape change.
9. **Add an end-to-end `relax_and_split_jax` parity test** with `reg_l0 > 0` (and again with `reg_l1 > 0`) over ≥2 outer iterations to lock the orchestrator-level integration.

The first three are the only items I would consider blocking for a
production release; the rest are coverage hardening.
