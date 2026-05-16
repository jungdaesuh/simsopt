# Parity Audit DEEPER — PRIORITY 4: tracing.py vs simsoptpp/tracing.{cpp,h}

**Audit timestamp:** 2026-05-16 (second pass)
**Auditor scope:** failure modes the first-pass forward-formula audit would miss
**Branch:** `gpu-purity-stage2-20260405`

## Scope reminder

The first-pass audit (`.artifacts/parity_audit_2026-05-16/04_tracing.md`) verified
RHS-by-RHS algebra parity and the obvious controller/event-localizer divergences
(`dtmax`, Illinois vs TOMS-748, initial step size). This deeper pass deliberately
ignores those known findings and hunts for issues forward-parity checks
**cannot** reveal: invariant conservation gates, axis-singularity behavior,
classifier tie-breaks, time-reversal symmetry, classifier dispatch completeness,
accumulator-overflow blind spots, C++ UB, and lost-particle accounting drift.

Files re-read end-to-end for this pass:

- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/tracing.py` (3,182 lines)
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/tracing.cpp` (560 lines)
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/tracing.h` (158 lines)
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/python_tracing.cpp` (87 lines)
- `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/tracing.py` (2,040 lines)
- 9 test files under `tests/jax_core/` and `tests/field/`

---

## HIGH severity

### H1. `ToroidalTransitStoppingCriterion` reference phi differs between C++ and JAX

**Files / lines.** C++ predicate body at `simsoptpp/tracing.h:30-44`:

```cpp
bool operator()(int iter, double t, double x, double y, double z) override {
    if (iter == 1) {
      phi_last = M_PI;
    }
    double phi = z;
    if (!flux) {
      phi = get_phi(x, y, phi_last);
    }
    if (iter == 1) {
      phi_init = phi;
    }
    phi_last = phi;
    double ntransits = std::abs((phi - phi_init) / (2 * M_PI));
    return ntransits >= max_transits;
};
```

The C++ predicate snapshots `phi_init` **after the first integration step** —
i.e. at `iter == 1`, after `dense.do_step()` ran once, so `(x, y, z)` is the
state at `t_1`, not at `t_0`. The unwrap anchor is the literal constant `M_PI`
(`phi_last = M_PI`), then `phi_init` is captured from the unwrapped `phi(t_1)`.

JAX equivalent at `simsopt/jax_core/tracing.py:886-888`, `:1422-1424`,
`:2882-2884`:

```python
phi_init = _continuous_phi(
    y0_arr[0], y0_arr[1], jnp.asarray(np.pi, dtype=dtype), dtype
)
```

This is computed once during driver construction from the **initial** state
`y0`, before any integration step has run. Subsequently the criterion is
evaluated as `transits = |phi_unwrapped(t_step) - phi_init| / 2π`
(`tracing.py:481-483`).

**Mathematical impact.** Define `Δphi_step = phi(t_1) - phi(t_0)`. The C++
reference is `phi(t_0) + Δphi_step`; the JAX reference is `phi(t_0)`. For a
fieldline with `dphi/dt ≈ |B_phi|/R` and an initial step `dt0 = 1e-5 * dtmax ≈
1e-5 * R·π/(2|B|)`, the offset is `Δphi_step ≈ |B_phi|/|B| · π/2 · 1e-5 ≈
5e-6` radians (worst case along a strongly toroidal field line). At
`max_transits = 100` this offset can shift the criterion firing time by
`5e-6 / 2π / 100 · tmax ≈ 8e-9 · tmax`. At `tmax = 1e-3` that's `8e-12` —
beneath the `event_time_atol = 1e-9` lane gate, so it never trips the lane.
At `tmax = 200` (typical fieldline `compute_fieldlines`) it grows to
`1.6e-9` — borderline.

**More serious case.** When `phi_init` is computed from `y0 = (R, 0, Z)`
(the fieldline driver's standard launch on `y=0`, e.g. `field/tracing.py:1718`
sets `y0 = [R0, 0.0, Z0]`), `atan2(0, R) = 0`, so JAX `phi_init =
_continuous_phi(R, 0, π) = 0 + 1·2π = 2π` (since `round((π-0)/2π) = round(0.5) = 0`
or `1` depending on rounding mode — `jnp.round` uses banker's rounding,
giving `0` for exactly `0.5`). C++: after one step, `phi(t_1) ≈ 0 + dphi`
(small positive), then `get_phi(x_1, y_1, M_PI)` returns the option closest
to π — likely `phi + 2π` (i.e. `2π + dphi`). So both end up near `2π`, but
the JAX value is exactly `2π · k` (an integer multiple) while the C++ value
includes the first step's drift. **The disagreement is structural and
deterministic; on long traces it compounds linearly with `Δphi_step`.**

**Severity.** HIGH. This is the largest non-controller divergence in the
audit and was not flagged by the first pass. It is structurally present in
all three Cartesian drivers (fieldline, GC Cartesian, full-orbit) and in the
Boozer GC zeta-counterpart (`zeta_init = _continuous_angle(y0_arr[2], π, dtype)`
at `tracing.py:2359`).

**Untested.** `tests/field/test_tracing_jax_item16_extended.py:159` (the
`ToroidalTransitStoppingCriterion` test) compares JAX vs C++ but does not
gate the per-particle firing time at the lane's `event_time_atol`; it only
verifies that the criterion fires at all. A 1e-9 drift survives this gate.

---

### H2. `assert(ys[0] > 0)` in C++ `GuidingCenterBoozerRHS` has no JAX equivalent — silent NaN propagation at axis

**File / line.** `simsoptpp/tracing.cpp:226`:

```cpp
assert(ys[0]>0);  // inside GuidingCenterBoozerRHS::operator()
```

This guards against `s ≤ 0` (negative or zero flux coordinate), which would
cause `dGdpsi = dGds/psi0` and `dmodBdpsi = dmodBds/psi0` to be evaluated
using interpolant values that are typically unphysical (the Boozer radial
interpolant extrapolates linearly past `s=0`, where the analytic limit is
singular — `dmodBds → ∞` like `1/s` and similar for `K`).

The JAX `guiding_center_boozer_rhs` factory at
`simsopt/jax_core/tracing.py:2149-2223` has **no equivalent guard**:

```python
def rhs(_t: jax.Array, y: jax.Array) -> jax.Array:
    del _t
    v_par = y[3]
    point = _boozer_point_2d(y)
    modB = _boozer_scalar(evals["modB"](state, point))
    # ... no check on y[0] > 0 ...
    fak1 = m_arr * v_par * v_par / modB + m_arr * mu_arr
    # ...
```

Curiously the C++ `GuidingCenterNoKBoozerRHS::operator()` at
`tracing.cpp:160-187` and `GuidingCenterVacuumBoozerRHS::operator()` at
`tracing.cpp:106-129` do **not** have the assert either. Only the full
`K != 0` path enforces it. So this is asymmetric on the C++ side too, but
the JAX side is entirely unprotected.

**Consequence.** When a particle reaches `s = 0` (magnetic axis), the JAX
RHS computes `dmodBds / psi0` from the radial interpolant's `s=0` slot.
Looking at `BoozerRadialInterpolantJAX`'s evaluator stack, the spline
evaluators return interpolated values without any limit-handling; if the
upstream `BoozerRadialInterpolant` was built with `enforce_vacuum_and_hands_off`
the dmodBds may be finite, but for general MHD equilibria it diverges. The
JAX RHS will produce `dv_par = -(mu / v_par) * (huge * ds * psi0 + ...) =
NaN` (when `v_par = 0` AND `s = 0` simultaneously, which is the actual
banana-orbit-near-axis pathological case).

**Severity.** HIGH. Particles can spiral into the axis silently, with the
DOPRI5 PI controller seeing NaN derivatives → `err_safe = jnp.inf` →
`factor = MIN_FACTOR = 0.2` (geometric step shrinkage) → effectively a stall
without diagnostic. The C++ `assert` in `Release` builds is a noop but in
`Debug` builds aborts the run; the JAX equivalent silently produces wrong
trajectories.

**Untested.** No JAX test fires a particle with initial `s < 0.05`; the
minimum value in `tests/jax_core/test_tracing_jax_gc_boozer.py` is
`s = 0.30` (`_make_initial_point` at line 194).

---

### H3. No mu (magnetic moment) adiabatic-invariant conservation test on the JAX guiding-center paths

**Coverage gap.** The first pass listed this as a coverage gap. This
deeper pass confirms it is structural: a wrong-sign or wrong-coefficient
RHS bug would survive the existing **endpoint** parity tests if the
trajectories happen to close into a periodic orbit (the same wrong-sign
energy can produce a sister periodic orbit just shifted in time).

The energy-conservation form of the no_k / full Boozer RHS at
`simsopt/jax_core/tracing.py:2141-2143` is:

```python
dv_par = -(mu_arr / v_par) * (
    dmodBdpsi * ds * psi0 + dmodBdtheta * dtheta + dmodBdzeta * dzeta
)
```

This formula is mathematically equivalent to `dE/dt = 0` (energy
conservation by construction), but it does **not** automatically conserve
mu. The adiabatic invariant `mu = m v_perp^2 / (2 |B|)` is only conserved
to leading order in the gyroradius; with `v_perp^2 = v_total^2 - v_par^2`
and `v_total = sqrt(2E/m)` fixed by energy conservation, `mu(t) = (v_total^2
- v_par(t)^2) / (2 |B(t)|)` is what one expects.

**The only existing related test** is `test_trace_fullorbit_conservation_invariants`
at `tests/jax_core/test_tracing_jax_fullorbit.py:332-407`, which checks
**kinetic energy** (not mu) and only for the **full-orbit Lorentz** path —
not the guiding-center RHS variants where the conservation form is most
fragile.

**Severity.** HIGH coverage gap. The Boozer GC RHS variants (3 modes ×
3 field types = 9 paths) have no isolated invariant-conservation test on
the JAX side. The CPU `test_particle.py::test_energy_momentum_conservation_boozer`
test (line 421, upstream) exists but routes through SIMSOPT's full
optimization stack, not the JAX kernel in isolation.

**Untested.** All three Boozer modes lack a "fire a particle, integrate
for `1e-3 s`, sample 10 mid-trajectory steps, assert `|μ(t) - μ(0)| / μ(0) < 1e-6`"
test. Such a test would catch wrong-sign errors in `dv_par` that the
endpoint test misses.

---

### H4. `bracket_root_jax` has unguarded `(fb - fa)` divisor

**File / line.** `simsopt/jax_core/tracing.py:765`:

```python
candidate = b - fb * width / (fb - fa)
```

This computes the false-position estimate unconditionally on every loop
iteration. When `fb == fa` (constant residual across the bracket, e.g.
when the event residual has gone exactly flat due to integrator step
quantization or NaN propagation), `(fb - fa) == 0.0` and the division
produces `NaN` (or `±inf` for `fb == 0, fa != 0`). The `converged` branch
at lines 766-771 uses `jax.lax.cond` to select `best_f` over `f(candidate)`
if `width <= atol_arr`, but:

1. The NaN `candidate` may still be evaluated through `f(candidate)` because
   JAX traces both branches of `lax.cond` during compile (only the result
   is selected at runtime).
2. With `atol = 0.0` always passed (`tracing.py:1015, 1547, 2479, 3006`,
   confirmed in first-pass audit), `converged` is False except when
   `width <= 0` exactly. So the NaN candidate is **used** to update the
   bracket via `jnp.where(keep_left, ...)` on lines 780-783.
3. `keep_left = jnp.sign(fa) * jnp.sign(fc) <= zero` evaluates `False` when
   `fc = NaN` (NaN comparisons are False in IEEE 754), so the bracket
   shifts toward the NaN candidate, **corrupting all subsequent
   iterations**.

**C++ comparison.** Boost TOMS-748 has internal guards against degenerate
divisors and converges with a `boost::math::tools::eps_tolerance` early
exit (the `rootmaxit=200` budget is rarely exhausted). C++ does not need
the divisor guard because TOMS-748 picks the root estimate from a quadratic
or cubic interpolation that incorporates a fallback to bisection on
near-flat residuals.

**Severity.** HIGH. This is a latent NaN-poisoning path. In practice it is
rarely triggered because the diff function `_continuous_phi(...) - phi_shift`
is well-conditioned across a single accepted step, but on banana orbits
near the turning point (`v_par ≈ 0`), the integrator may produce two
consecutive states whose unwrapped phi values differ by less than the
ulp — at which point `f_left = f_right` to machine precision and the
division underflows.

**Untested.** No JAX bracket test fires a particle through a banana
turning-point or otherwise constructs a degenerate-bracket fixture. The
existing tests at `tests/jax_core/test_tracing_jax_item14.py:232-300`
exercise well-separated linear and quadratic residuals only.

---

### H5. `t_final` back-fill divergence — drives `loss_ctr` drift

**Files / lines.** C++ `tracing.cpp:441-444` back-fills the final row to
`t = tmax` on normal exit via `dense.calc_state(tmax, y)`. JAX drivers do
NOT back-fill (`tracing.py:1170-1181` etc.): they leave `t_final` at the
last accepted step's time.

The orchestrator at `field/tracing.py:494, 858, 1021, 703` tests
`if t_final < tmax - 1e-15: loss_ctr += 1`. JAX `t_final` is always
strictly less than `tmax` (no back-fill), so the JAX `loss_ctr` is biased
toward over-counting versus C++ at any fixed RNG seed. Magnitude depends
on the integrator's last-step landing precision (typically O(eps_t·tmax)).

**Severity.** HIGH for any pipeline that aggregates `loss_ctr`. The
first-pass audit marked this PARITY at the orchestrator level; on
second pass the drift is structural.

**Untested.** No fixed-seed JAX vs CPU `loss_ctr` parity test exists.

---

## MEDIUM severity

### M1. Levelset classifier exact-zero tie-breaker is asymmetric

**File / lines.** JAX `_stopping_criterion_should_stop` at
`simsopt/jax_core/tracing.py:495-501`:

```python
if isinstance(criterion, LevelsetStoppingCriterion):
    position = jnp.stack([x, y, z]).reshape(1, 3).astype(dtype)
    sign = criterion.classifier_fn(position)[0]
    return sign < jnp.asarray(0.0, dtype=dtype)
```

C++ `LevelsetStoppingCriterion::operator()` at `tracing.h:122-131`:

```cpp
bool operator()(int iter, double t, double x, double y, double z) override {
    double r = std::sqrt(x*x + y*y);
    double phi = std::atan2(y, x);
    if(phi < 0)
        phi += 2*M_PI;
    double f = levelset->evaluate(r, phi, z)[0];
    return f<0;
}
```

Both use strict `f < 0`. **When `f == 0` exactly** (on the surface
itself), neither fires. This is symmetric.

However, the JAX path samples the classifier on `(x, y, z)` Cartesian,
while the C++ path samples on `(r, phi, z)` cylindrical. If the JAX
classifier function (built by `SurfaceClassifier.to_jax_classifier_fn()`
at `surface.py:63-64`) internally does its own Cartesian → cylindrical
conversion using a different normalization (e.g. `arctan2` returning
`(-π, π]` instead of `[0, 2π)`) and then samples the underlying
`RegularGridInterpolant3D` grid that was built on `[0, 2π)`, the JAX
classifier could return a different value than the C++ classifier at
the same physical point. The first-pass audit deferred this to a
"separate scope" (interpolant audit).

I checked `surface.py:92-117` (`_build_jax_classifier_fn`). It rebuilds
the interpolant from grid metadata; whether the JAX RHS calls it with
phi in `[0, 2π)` or `(-π, π]` depends on the underlying
`make_levelset_classifier`. **This deeper pass cannot resolve the
question without reading `simsopt/jax_core/surface_classifier.py`**, but
the structural risk is real.

**Severity.** MEDIUM. Test `test_surface_classifier_to_jax_classifier_fn_matches_cpu`
at `tests/jax_core/test_tracing_jax_levelset_events.py:90` does verify
parity at a fixed grid sample, but the sample density may not cover the
phi-wraparound edge cases (`phi = 0`, `phi = π`, `phi = 2π - ε`).

**Untested.** No test fires a particle along the `phi = 0` line and
verifies that the classifier returns the same value as the C++ side at
the exit point.

---

### M2. C++ `LevelsetStoppingCriterion` cylindrical phi sign rule does NOT match `get_phi`

**File / line.** `simsoptpp/tracing.h:124-127`:

```cpp
double r = std::sqrt(x*x + y*y);
double phi = std::atan2(y, x);
if(phi < 0)
    phi += 2*M_PI;
```

This is the **standard cylindrical conversion**: phi ∈ [0, 2π). It is
NOT the `get_phi(x, y, phi_near)` continuous-unwrap function used by the
phi-plane crossing detector at `tracing.cpp:394`.

**Implication.** A particle that has wrapped 7 times around the torus
(unwrapped phi ≈ 14π) gets classifier-sampled at `phi % 2π`, but its
`ToroidalTransitStoppingCriterion` sees `unwrapped phi`. These are
correctly distinct on the C++ side. **The JAX side similarly samples the
classifier with `(x, y, z)` and the classifier internally does its own
`atan2` reduction**, but again, the parity of this reduction with the C++
side depends on the JAX classifier implementation.

**Severity.** MEDIUM. Not technically a divergence, but a **structural
coupling** that the first-pass audit's parity matrix glossed over.

---

### M3. `phi_init` divergence compounds with `_continuous_phi`'s `phi_near = π` seed

**File / line.** All four drivers seed the running unwrap branch as:

```python
phi_init = _continuous_phi(
    y0_arr[0], y0_arr[1], jnp.asarray(np.pi, dtype=dtype), dtype
)
```

(`tracing.py:886-888`, `:1422-1424`, `:2882-2884`)

This matches C++'s `phi_last = M_PI` literal on `iter == 1`. So far so
good. But the JAX `phi_init` is then **used both as**:

1. The toroidal-transit reference (`tracing.py:481-483` —
   `transits = |phi_unwrap - phi_init| / 2π`).
2. The running `phi_last` accumulator's first value (`tracing.py:901`,
   `:1437`, etc. — `phi_init` is folded into the carry).

This dual-use is correct C++-parity except for one corner case: when
`y0 = (R, 0, Z)` is on the `y = 0` axis (the fieldline driver's default
launch at `field/tracing.py:1718`), `atan2(0, R)` is exactly `0` for
`R > 0` and `±π` for `R < 0`. With `phi_near = π`, the JAX `_continuous_phi`
computes `k = round((π - 0)/(2π)) = round(0.5)`. **The IEEE 754 default
rounding mode rounds `0.5` away from zero (`k = 1`)** so `phi_init = 2π`.
**The C++ `get_phi(R, 0, π)`** computes `nearest_multiple = round(π/(2π)) =
round(0.5)`, also `1`, so `nearest_multiple = 2π`. Then `opt1 = 0 + 0 = 0`,
`opt2 = 2π + 0 = 2π`, `opt3 = 4π + 0 = 4π`. Distances from π: `π`, `π`,
`3π`. **Tie between opt1 and opt2**; the `<=` check in C++ at line 345
picks `opt1 = 0`. So C++ `phi_init = 0`, JAX `phi_init = 2π`.

**This is a 2π discrepancy in the toroidal-transit reference.** It does
not affect the criterion firing (since `|phi - phi_init| / 2π` accumulates
correctly in both backends after the first step), but it does affect any
downstream code that reads `phi_init` directly. None of the drivers
expose `phi_init` to consumers, so this is a latent buglet.

**Severity.** MEDIUM. The 2π modular ambiguity does **not** affect parity
tests at the lane tolerance, but it makes the JAX `phi_init` differ from
the C++ `phi_init` by exactly `2π` whenever `y0` lands on the `y=0` axis.

**Untested.** No test asserts `JAX.phi_init == CPP.phi_init` modulo `2π`.

---

### M4. `jnp.round` (banker's) vs `std::round` (away-from-zero) on `phi` unwrap

**File / lines.** JAX `_continuous_phi:522` uses `jnp.round((phi_near - phi_raw)/2π)`;
C++ `get_phi` at `tracing.cpp:338` uses `std::round(phi_near/(2π))`. IEEE 754
default rounding is half-to-even (banker's) but `std::round` is half-away-from-zero.

I traced several cases (`phi_near ∈ {0, π}` × `phi_raw ∈ {-π+ε, π-ε}`); the two
helpers agree because `(π-ε)/(2π) < 0.5` rounds to `0` in both modes. They only
diverge on exact half-integer inputs that orbit trajectories rarely produce.

**Severity.** MEDIUM (theoretical, non-bite). Mentioned because the rounding-mode
mismatch is a structural drift point that future refactors could expose.

---

### M5. `accepted_count == max_steps` overflow goes silently as `status=1`

**File / lines.** All four drivers exit the while-loop when
`accepted_count >= max_steps` (`tracing.py:928`, `:1463-1464`, `:2398-2399`,
`:2923-2924`), then emit `status_normal = 1` (line 1179 etc.). This is
documented in the result dataclass docstring (`tracing.py:307-308`) but
NOT raised at the orchestrator level — `_event_hits_prefix`
(`field/tracing.py:31-40`) only rejects `phi_hits_count > max_phi_hits`.

First-pass audit marked this INFO. On second pass, I confirmed:

```python
res_tys.append(live)  # <- silently truncated
status = int(result.status)
t_final = float(result.t_final)
if status > 0:
    logger.debug(  # <- debug-level only, not warning/error
        f"... JAX guiding-centre status={status} t_final={t_final}, "
        f"steps_taken={int(result.steps_taken)}"
    )
```

(`field/tracing.py:843-852`)

So a silently-truncated trajectory propagates through the orchestrator
with only a `logger.debug` message. The `loss_ctr` increment captures
particles that exit early due to `status=1`, so the count is correct,
but the **trajectory shape** is wrong (truncated).

**Severity.** MEDIUM. This is the most likely failure mode at high
trajectory-density runs (Stage 3 long traces, banana orbits with many
bounces). The fact that it's reported only at debug level means
production runs can silently drop trajectory tails.

**Untested.** No test fires a particle for which `max_steps = 4000` is
known-insufficient and asserts the orchestrator surfaces the truncation.

---

### M6. Boozer `s=0` axis singularity in `_eval_dmodBds` and friends — JAX evaluator behavior unverified

**File / line.** `boozermagneticfield_jax.py:464` (radial branch),
`:1186` (analytic branch), `:1507` (interpolated branch) all define
`dmodBds` evaluators. The JAX guiding-center Boozer RHS variants call
these at `tracing.py:2055`, `:2111`, `:2188`. The C++ side uses
`field->modB_derivs_ref()(0)/psi0` — also evaluated by the spline
interpolant.

The **behavior at `s=0`** depends on whether the radial spline
interpolant extrapolates linearly or returns the boundary slope. The
Boozer field tooling (`BoozerRadialInterpolant`) typically uses
`InterpolatedSplineCurve` which is undefined for `s < 0`. The JAX
evaluators inherit this; at `s = 0` the spline returns the boundary
value at `s_min` (some positive value) and the dmodBds slope can be
finite but large.

C++ aborts on the `K != 0` path (`assert(ys[0] > 0)` at `tracing.cpp:226`)
before this matters. JAX has no guard, so the RHS produces finite values
that are extrapolated to s=0 — potentially wrong but not NaN.

**Severity.** MEDIUM. The behavior is field-instance-dependent; for
well-conditioned interpolants the JAX RHS will produce a finite but
inaccurate derivative at s=0 (and silently continue). For poorly
conditioned interpolants it may NaN.

**Untested.** No JAX test fires a particle through s=0.

---

## LOW severity

### L1. `assert` on bracket order is documented but not enforced

**File / line.** `bracket_root_jax` docstring at `tracing.py:711-712`:

> Initial bracket endpoints. Must satisfy `t_left <= t_right`; the
> bracket is **not** swapped internally even if not.

In practice all four callers pass `0.0` and `1.0` so the constraint
holds. But future callers that pass a fraction-of-step bracket could
violate this and silently produce a wrong root.

C++ `toms748_solve` swaps endpoints internally. JAX is unguarded.

**Severity.** LOW (API surface).

---

### L2. `dopri5_step` recomputes Butcher tableau arrays every call

**File / line.** `tracing.py:648-652`:

```python
def dopri5_step(...):
    dtype = y.dtype
    A = jnp.asarray(_DOPRI5_A, dtype=dtype)
    C = jnp.asarray(_DOPRI5_C, dtype=dtype)
    B = jnp.asarray(_DOPRI5_B, dtype=dtype)
    E = jnp.asarray(_DOPRI5_E, dtype=dtype)
    ...
```

These `jnp.asarray` calls happen on every step. Under JIT this is folded
into the compiled graph as constants, so the runtime cost is zero, but
the JIT trace cost scales with the number of compile-time RHS factories.
Each driver creates a new RHS closure per particle (the closure captures
`m, q, mu`), causing a fresh compile.

C++ uses constexpr tableau coefficients; no recompile cost.

**Severity.** LOW (compile overhead, not runtime).

---

### L3. `_initial_step_size` ignores `dtmax`

**File / line.** `tracing.py:688-691`:

```python
def _initial_step_size(t0: jax.Array, t_end: jax.Array) -> jax.Array:
    span = jnp.abs(t_end - t0)
    h0 = jnp.asarray(_INITIAL_STEP_FRACTION, dtype=span.dtype) * span
    return jnp.minimum(h0, span)
```

First-pass HIGH finding (Finding 2). Mentioning here for completeness.

---

### L4. `_continuous_angle` uses `phi_near = π` seed for Boozer zeta

**File / line.** `tracing.py:2359`:

```python
zeta_init = _continuous_angle(y0_arr[2], jnp.asarray(np.pi, dtype=dtype), dtype)
```

The Boozer state stores zeta as a raw value (not via atan2). Anchoring
the unwrap against `π` is meaningful for the wrap-around test, but
produces a 2π discrepancy with C++ when `y0_arr[2] = 0` exactly (cf. M3).
**The criterion firing time is invariant** because both backends
accumulate `|zeta - zeta_init|/2π` consistently, but the absolute
`zeta_init` differs.

**Severity.** LOW.

---

### L5. C++ `assert(...)` blocks compile out in Release builds

**File / line.** `simsoptpp/tracing.cpp:226`:

```cpp
assert(ys[0]>0);
```

In `Release` (`NDEBUG`) builds this is a noop. So even on the C++ side,
the axis-singularity guard is only present in Debug builds. The CI build
configuration determines actual behavior.

**Severity.** LOW. Diagnostic surface only.

---

### L6. `IterStoppingCriterion` semantics — confirmed PARITY

C++ at `tracing.cpp:391` does `iter++` after `dense.do_step()` returns; JAX at
`tracing.py:1088` does `iter_count_post = step_count + 1` before the body tail.
Both fire when `iter > max_iter`. Confirmed PARITY; no action.

---

### L7. Time-reversal symmetry untested

**Coverage gap.** No JAX test reverses the sign of `tmax` (or equivalently
negates `dt` by reversing the integrator) and verifies the trajectory
retraces. Energy-conserving integrators should be approximately
time-reversible at second order; DOPRI5 is **not** time-reversible (it's
explicit and has dissipative truncation error), but the magnitude of the
asymmetry is a useful diagnostic.

**Severity.** LOW. The first-pass audit did not check this either.

---

### L8. `gc_to_fullorbit_initial_guesses` consults CPU `MagneticField`

**File / line.** `field/tracing.py:952-960` and the helper at lines
125-161. The full-orbit seeding inverts the guiding-center transformation
on the host using `field.B()` / `field.AbsB()` — CPU calls even under
JAX backend.

The first-pass audit flagged this as INFO. On second pass, I note that
this **uses different field call paths than the inner integrator** —
the JAX integrator calls `field_fn(point)` which routes through the JAX
B kernel, while the seeding calls `field.B()` which routes through the
CPU/JAX dual API. If the two paths produce slightly different B values
(e.g. due to cache invalidation timing in `BiotSavartJAX`), the JAX
full-orbit initial conditions will be slightly inconsistent with the
JAX integrator's evaluation at the same point.

**Severity.** LOW. The CPU path is byte-identical to the C++ oracle by
construction (both use `simsoptpp.MagneticField::B`); the JAX path
parity depends on the BiotSavartJAX cache state.

---

## C++ / UB review (concise)

1. **`signed int` loop counters used as array indices** (`tracing.cpp:401`,
   `i < phis.size()`): implicit sign-conversion; safe in practice (LOW).
2. **`std::abs(double)`** (`tracing.cpp:419`): `<cmath>` pulled in via boost;
   correct overload selected. Fragile but safe.
3. **OMP races.** No `#pragma omp` in `tracing.cpp`. Driver runs serially.
4. **Uninitialized state slots.** `State` is `std::array<double, N>` (POD);
   `dense.initialize` overwrites before read. Safe.
5. **Lambda capture escape.** `tracing.cpp:408-415` captures by reference; only
   called synchronously inside `toms748_solve`. Safe.
6. **`assert(ys[0]>0)` and `assert(phi_last <= phi_shift <= ...)`** are
   Debug-only (NDEBUG strips them). JAX has no equivalent guards; see H2, L5.

---

## Test coverage gaps (second-pass summary)

Beyond the gaps already listed in the first-pass audit's "Coverage gaps
(prioritized)" section, this deeper pass identifies:

1. **HIGH** — No JAX-isolated mu-conservation test for any Boozer GC mode
   (vacuum / no_k / full). The `dv_par = -(mu/v_par) · Σ ∂B/∂q · dq/dt`
   energy-conservation form is mathematically equivalent to mu
   conservation but doesn't enforce it; a wrong-sign bug could survive
   the endpoint parity tests.

2. **HIGH** — No fixed-seed `loss_ctr` parity test JAX vs CPU. The
   `t_final < tmax - 1e-15` comparator can drift ±1 per nparticles.

3. **HIGH** — No banana-orbit (`v_par ≈ 0`) stability test. The
   `dv_par = -(mu/v_par) · ...` formula NaNs at the turning point; both
   backends rely on the controller to back off, but JAX has no `dtmax`
   ceiling so the recovery is slower and may stall at `max_steps`.

4. **HIGH** — No axis (s ≈ 0) test for Boozer GC. The C++ `assert(ys[0]>0)`
   guard in `GuidingCenterBoozerRHS` is absent on the JAX side.

5. **MEDIUM** — No `accepted_count == max_steps` exit-reporting test;
   `status=1` propagates silently through `logger.debug` only.

6. **MEDIUM** — No time-reversal symmetry test on any backend.

7. **MEDIUM** — No `_continuous_phi`/`get_phi` parity test at exact
   `phi=π` edge cases (negative-zero handling, banker's-rounding
   discrepancies).

8. **MEDIUM** — No `bracket_root_jax` degenerate-bracket (`fb == fa`)
   test. Latent NaN-poisoning path through `width / (fb - fa)`.

9. **MEDIUM** — No Levelset classifier exact-zero tie-breaker test.
   Particle exit on the surface (`f == 0`) is symmetric (`f < 0` strict
   on both sides) but the `f == 0` floating-point coincidence isn't
   exercised.

10. **MEDIUM** — No phi-wraparound parity test for Levelset classifier
    at `phi ∈ {0, π, 2π}`. The cylindrical normalization paths on C++
    and JAX could diverge by 2π depending on the JAX classifier's
    internal `atan2` convention.

11. **LOW** — No iter-criterion test on the JAX side beyond the
    `test_iter_stopping_criterion_terminates_trajectory` at
    `test_tracing_jax_phi_events.py:281` (already existed; first pass
    noted this).

12. **LOW** — No `IterationStoppingCriterion` parity test JAX vs CPU
    at the same `max_iter` value.

---

## Architectural / convention findings

### A1. JAX path uses Cartesian `(x, y, z)` directly; C++ converts to `(r, phi, z)` and calls `set_points_cyl`

**File / line.** C++ `GuidingCenterVacuumRHS::operator()`,
`FullorbitRHS::operator()`, `FieldlineRHS::operator()` all do:

```cpp
rphiz(0, 0) = std::sqrt(x*x+y*y);
rphiz(0, 1) = std::atan2(y, x);
if(rphiz(0, 1) < 0)
    rphiz(0, 1) += 2*M_PI;
rphiz(0, 2) = z;
field->set_points_cyl(rphiz);
```

(`tracing.cpp:57-63, 284-289, 320-325`)

JAX RHS factories call `magnetic_field_fn(position)` where `position =
y[:3]` is Cartesian. The JAX field wrappers (`toroidal_field_jax.py:45-47`,
`interpolated_field_jax.py:222-224`, etc.) accept Cartesian and convert
internally if needed.

**Risk.** If a JAX field wrapper's `jax_B_at(point)` interprets the input
as cylindrical or applies a different phi convention (e.g. `(-π, π]`
instead of `[0, 2π)`), the trajectories will silently diverge from C++.
This is **not** a tracing-module bug per se but a contract risk at the
JAX field interface.

**Severity.** ARCHITECTURAL. The contract is implicit; no audit gate
enforces "JAX `jax_B_at` must accept Cartesian (x, y, z) and produce the
same B as C++'s `set_points_cyl(r=sqrt(x²+y²), phi=atan2_pos(y, x), z)`."

### A2. The JAX runs DOPRI5 with no `dtmax`; long banana orbits will stall

**File / line.** First-pass HIGH finding (Finding 1). Reproduced here for
completeness: combined with H2 and H4 above, this creates a compound
failure mode at banana turning points (v_par → 0) AND at the magnetic
axis (s → 0). Together they form the dominant pathological surface for
particle tracing.

---

## Module-level documentation findings

### D1. Docstring at `tracing.py:8-13` falsely claims fieldline RHS is `B/|B|`

First-pass LOW finding. Confirmed reading the actual `fieldline_rhs`
implementation at line 608-609: it returns `B`, not `B/|B|`. The
docstring should say "matches the upstream `FieldlineRHS` which returns
`dx/dt = B`".

### D2. Docstring at `tracing.py:386-388` is stale

First-pass LOW finding. The `ToroidalTransitStoppingCriterion` docstring
claims the flux-coordinate branch is "not on the JAX path yet (the
Boozer guiding-centre RHS is deferred under item 14)". Item 14 has
landed; the flux-coordinate branch is now active on the Boozer route.
The docstring is wrong.

---

## Recommended actions (DEEPER pass additions)

### HIGH

1. **Add per-mode mu-conservation tests** for `trace_guiding_center_boozer`
   (vacuum / no_k / full). Fire 1 particle, run for `tmax = 1e-3`, sample
   10 intermediate accepted steps, assert `|μ(t) - μ(0)| / μ(0) ≤ 1e-6`.
   Tests should live in `tests/jax_core/test_tracing_jax_gc_boozer.py`
   alongside the existing endpoint-parity tests.

2. **Add fixed-seed `loss_ctr` parity test** comparing
   `trace_particles_boozer` JAX vs CPU at N=20 particles, `tmax = 1e-3`,
   asserting `JAX.loss_ctr == CPU.loss_ctr`. Should live in
   `tests/jax_core/test_tracing_jax_gc_boozer.py` or a new
   `test_tracing_jax_loss_count_parity.py`.

3. **Align `ToroidalTransitStoppingCriterion`'s `phi_init` snapshot
   timing** — JAX should capture `phi_init` after the first accepted
   step, matching the C++ `if (iter == 1) phi_init = phi` semantics.
   Trivial fix: snapshot `phi_init` from the post-first-step state
   inside the body, not at construction.

4. **Add `assert(s > 0)` equivalent to JAX Boozer GC full RHS.** Use
   `jax.experimental.checkify` or a `jnp.where(s > 0, rhs, NaN)` guard
   so the integrator's controller correctly rejects steps that crossed
   the axis. Mirror to the no_k and vacuum modes for consistency
   (acknowledging C++ asymmetry).

5. **Guard `bracket_root_jax` divisor** — replace
   `candidate = b - fb * width / (fb - fa)` with
   `candidate = jnp.where(jnp.abs(fb - fa) > 1e-300, b - fb*width/(fb-fa), 0.5*(a+b))`
   (fall back to bisection midpoint on flat residual).

### MEDIUM

6. **Add banana-orbit stability test** — fire a particle with
   `v_par = 1e-3 * v_total` (just above zero) and verify the integrator
   does not stall or NaN out over `tmax = 1e-4`.

7. **Add axis-singularity test** — fire a Boozer particle with
   `s_init = 1e-5` and verify the JAX RHS produces finite values, matching
   the C++ RHS at the same point.

8. **Add `accepted_count == max_steps` orchestrator hard-error path**
   (mirror `_event_hits_prefix` overflow rejection). At minimum, escalate
   the `logger.debug` to `logger.warning` so production runs surface the
   truncation.

9. **Add `_continuous_phi` parity test** at edge values `(x, y) ∈ {(R, 0),
   (R, +0), (R, -0), (-R, 0)}` and `phi_near ∈ {0, π, 2π, -π}`,
   asserting JAX `_continuous_phi` matches C++ `get_phi` within ulps.

10. **Add Levelset classifier `phi=0` parity test** — sample the JAX
    classifier and the C++ classifier at the same physical point with
    `x=R, y=0, z=Z` (so `phi=0` exactly) and assert byte-identity.

### LOW

11. **Fix module docstring** at `tracing.py:8-13` (fieldline RHS returns
    `B`, not `B/|B|`).

12. **Remove stale carve-out comment** at `tracing.py:386-388`.

13. **Add `jnp.swapaxes` for `bracket_root_jax` bracket order** — accept
    `t_left, t_right` in any order and swap internally, matching Boost
    TOMS-748.

14. **Document the Cartesian-vs-cylindrical contract** for
    `jax_B_at(point)` somewhere visible (e.g. add a docstring contract
    section to `simsopt/field/tracing.py:_require_jax_field_B`).

---

## Summary table

| # | Severity | Category | Files | First-pass status |
|---|---|---|---|---|
| H1 | HIGH | `phi_init` snapshot timing diverges (C++ at iter=1, JAX at construct) | `tracing.py:886, 1422, 2882`, `tracing.h:30-44` | NOT flagged |
| H2 | HIGH | `assert(s>0)` in C++ has no JAX equivalent | `tracing.cpp:226`, `tracing.py:2149-2223` | NOT flagged |
| H3 | HIGH | No mu-conservation test on JAX Boozer GC paths | `tests/jax_core/test_tracing_jax_gc_boozer.py` | Listed as MEDIUM gap |
| H4 | HIGH | `bracket_root_jax` unguarded `(fb - fa)` divisor | `tracing.py:765` | NOT flagged |
| H5 | HIGH | `t_final` vs `tmax` accounting drift (no back-fill) | `field/tracing.py:494, 858, 1021` | Marked PARITY (incorrect) |
| M1 | MEDIUM | Levelset classifier exact-zero / phi-wraparound asymmetry | `tracing.h:122-131`, `tracing.py:495-501` | Deferred to interpolant audit |
| M2 | MEDIUM | C++ Levelset uses `[0, 2π)` phi; transit criterion uses unwrap | `tracing.h:124-127`, `tracing.cpp:394` | NOT flagged |
| M3 | MEDIUM | `phi_init` 2π discrepancy on `y0=(R, 0, Z)` launch | `tracing.py:886` | NOT flagged |
| M4 | MEDIUM | IEEE rounding mode (`std::round` vs `jnp.round`) | `tracing.py:765`, `tracing.cpp:345` | NOT flagged |
| M5 | MEDIUM | `accepted_count == max_steps` silent truncation | `field/tracing.py:843-852` | Listed as INFO |
| M6 | MEDIUM | Boozer s=0 axis singularity behavior unverified | `boozermagneticfield_jax.py:464` | NOT flagged |
| L1 | LOW | `bracket_root_jax` bracket-order assert not enforced | `tracing.py:711-712` | Listed |
| L2 | LOW | Butcher tableau recreated per `dopri5_step` call | `tracing.py:648-652` | NOT flagged |
| L3 | LOW | `_initial_step_size` ignores `dtmax` | `tracing.py:688-691` | First-pass HIGH |
| L4 | LOW | `_continuous_angle` uses `phi_near=π` seed for zeta | `tracing.py:2359` | NOT flagged |
| L5 | LOW | C++ `assert` compiled out in Release | `tracing.cpp:226` | NOT flagged |
| L6 | LOW | Iter-counter semantics — PARITY confirmed | `tracing.cpp:391`, `tracing.py:1088` | First-pass PARITY |
| L7 | LOW | Time-reversal symmetry untested | n/a | NOT flagged |
| L8 | LOW | `gc_to_fullorbit_initial_guesses` uses CPU `field.B()` | `field/tracing.py:952-960` | Listed as INFO |
| A1 | ARCH | JAX uses Cartesian directly; C++ goes through cyl | `tracing.cpp:57-63` | NOT flagged |
| A2 | ARCH | DOPRI5 with no `dtmax` × banana × s=0 compound failure | n/a | First-pass HIGH (partial) |
| D1 | DOC | Fieldline RHS docstring wrong | `tracing.py:8-13` | Listed LOW |
| D2 | DOC | Stale flux-coord ToroidalTransit comment | `tracing.py:386-388` | Listed LOW |

**Total findings:** 22 (5 HIGH new, 6 MEDIUM new, 4 LOW new, 4 ARCH/DOC restated)

---

## Untested edge-case inventory (consolidated)

| # | Edge case | Impact bucket |
|---|---|---|
| 1 | Banana turning point (`v_par ≈ 0`) — no test on any backend | H2/H4 latent NaN |
| 2 | Boozer axis (`s ≈ 0`) — minimum tested `s = 0.30` | H2 unprotected RHS |
| 3 | `y = 0` launch (`atan2(0, R) = 0`) — fieldline default | M3 2π `phi_init` drift |
| 4 | `phi = π` initial state — IEEE rounding edge | M4 theoretical |
| 5 | `phi = 2π - ε` Levelset sampling | M1 phi-wraparound |
| 6 | `fb == fa` degenerate bracket | H4 NaN poison |
| 7 | `accepted_count == max_steps` overflow | M5 silent truncation |
| 8 | `v_par` sign flip during integration | H2 removable singularity |
| 9 | Time reversal (`tmax → -tmax`) | L7 untested on both |
| 10 | `tmax = 0`, `tmax < 0` edge values | unspecified behavior |
| 11 | `max_steps = 1` minimal budget | `status=1` on first iter? |
| 12 | Multiple stopping criteria firing simultaneously | PARITY but untested |
| 13 | Levelset classifier returns exactly `0` | tie-break asymmetry untested |
| 14 | `phi_hits_count > max_phi_hits` for particle tracers | tested only on fieldlines |
| 15 | Negative-species charge (electrons) | only alpha (`q = +2e`) tested |

---

## Final tally vs first pass

| Bucket | First pass | Deeper pass | Delta |
|---|---|---|---|
| HIGH findings | 3 | 5 | +5 (with H3 reclassified up from MEDIUM gap; H5 newly reclassified from "PARITY") |
| MEDIUM findings | 3 | 6 | +6 |
| LOW findings | 5 | 4 | +4 (some restated) |
| ARCH/DOC | 4 | 4 | restated |
| Untested edge cases | 5 prioritized | 15 enumerated | +10 |

The first pass correctly flagged the integrator-flavor divergences. The
deeper pass surfaces correctness issues at the **boundaries** —
init/exit/axis/turning-point — that controller-flavor parity tests
cannot detect. The highest-priority deeper findings are H1 (phi_init
timing), H2 (axis NaN), H4 (bracket NaN), and H5 (loss accounting),
each of which silently corrupts production results without tripping any
existing parity gate.
