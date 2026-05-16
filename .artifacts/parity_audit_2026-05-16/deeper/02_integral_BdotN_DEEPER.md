# Parity Audit (Deeper Pass) 02 - `integral_BdotN` & SquaredFluxJAX

**Audit timestamp:** 2026-05-16 (second-pass / hunt-for-misses)
**JAX SSOT versions:** `jax==0.10.0`, `jaxlib==0.10.0`
**Branch:** `gpu-purity-stage2-20260405`

## What's in scope for the deeper pass

The first-pass audit (`.artifacts/parity_audit_2026-05-16/02_integral_BdotN.md`)
checked the three definitions' forward formulas against the C++ oracle and
against a NumPy reference, validated empirically that all three return finite
values that agree to `direct_kernel` tolerance, and flagged the
`"normalized"` denominator reduction-order divergence (M-1) plus the JAX-only
`residual_BdotN`/`signed_BdotN_flux` symbols having no C++ counterpart (I-1).

This deeper pass is explicitly looking for what forward parity sweeps would
**miss**: untested boundary inputs, gradient pathology at singular configs,
silent dtype drift, JIT-closure staleness, kwargs / error-class skew between
the JAX adapter and the CPU reference, OpenMP & broadcast hazards in C++ /
JAX, and shape-validation differences. I emitted small standalone probes
against the live `.conda/jax` env (`jax==0.10.0`) and captured concrete
input -> output examples for each finding.

## Top-line verdict

| Severity | Count |
|----------|-------|
| BLOCKER  | 0 |
| HIGH     | 3 |
| MEDIUM   | 4 |
| LOW      | 6 |
| INFO     | 3 |

The three HIGH findings are: (a) silent broadcast of mis-shaped `target` to
the wrong result (JAX) vs `RuntimeError` (C++); (b) silent NaN gradient on
the `"local"` path when one quadrature point has `|B|=0` (JAX returns
`val=inf` but a *finite* gradient on the surviving points, materially
different from `SquaredFlux.dJ()` which raises `ObjectiveFailure`); (c) the
`SquaredFluxJAX._raise_if_surface_dofs_drifted` guard has **zero test
coverage**, despite the docstring promise that surface mutation will be
caught. Each is described in detail below with the probe input / output.

## Files audited

| File | Role | Lines |
|------|------|-------|
| `src/simsopt/objectives/integral_bdotn_jax.py` | JAX kernel | 128 |
| `src/simsoptpp/integral_BdotN.cpp` | C++ reducer | 123 |
| `src/simsoptpp/integral_BdotN.h` | Declaration | 4 |
| `src/simsopt/objectives/fluxobjective_jax.py` | `SquaredFluxJAX` adapter | 433 |
| `src/simsopt/objectives/fluxobjective.py` | CPU `SquaredFlux` reference | 134 |
| `src/simsopt/jax_core/objectives_flux.py` | Spec-based JAX glue | 138 |
| `src/simsopt/jax_core/reductions.py` | Pairwise/Kahan/vdot baselines | 121 |
| `src/simsopt/jax_core/specs.py:429-444, 1397-1412` | `FixedSurfaceFluxSpec` constructor | (excerpt) |
| `src/simsopt/jax_core/_math_utils.py:30-69` | `as_jax_float64` (always upcasts) | (excerpt) |
| `tests/objectives/test_integral_bdotn_jax.py` | Kernel tests (def + C++ parity) | 454 |
| `tests/objectives/test_integral_bdotn_item10_closeout.py` | Chained `BS -> integral` parity at production scale | 247 |
| `tests/integration/test_stage2_jax.py` | Adapter parity, drift guards | 6473 |

---

## Findings

### HIGH-1 - Broadcast-silent acceptance of mis-shaped `target` in JAX

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:50`
**Counterpart:** `src/simsoptpp/integral_BdotN.cpp:34-43`

The C++ path validates `Btarget.shape(0) == nphi`, `Btarget.shape(1) == ntheta`,
and `Btarget.size() == nphi*ntheta` and raises `RuntimeError` on mismatch.
The JAX kernel does:

```python
# integral_bdotn_jax.py:42-51
nphi, ntheta, _ = Bcoil.shape
...
BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target   # broadcasts
```

`jnp.sum(Bcoil*unit_n, axis=-1)` has shape `(nphi, ntheta)`. Subtracting a
`(5,7)` `target` against a `(1,1,3)` `Bcoil` broadcasts to `(5,7)`, and the
final `pairwise_sum_flat` happily sums 35 entries divided by `nphi*ntheta=1`.

Reproduction:
```
B = jnp.ones((1,1,3)).at[..., 2].set(1.0)
n = jnp.zeros((1,1,3)).at[..., 2].set(1.0)
target_wrong = jnp.zeros((5,7))
integral_BdotN(B, target_wrong, n, "quadratic flux") -> 17.5  # WRONG, no error
sopp.integral_BdotN(np.asarray(B), np.asarray(target_wrong), np.asarray(n), ...) -> RuntimeError("Btarget has wrong shape.")
```

Same hazard exists for `(3,2)`-vs-`(2,3)` transposed targets at multi-point
shapes (raises `TypeError` from broadcasting later in the chain), and for
scalar targets (silently treated as a uniform offset). A scalar `target=0.0`
arguably is desired; a `(5,7)` target absolutely is not.

**Severity rationale:** A regression in a downstream caller that happens to
reshape a target buffer would silently produce wrong objective values, with
no test catching it. The CPU oracle protects against this; the JAX kernel
does not. **HIGH.**

**Fix sketch:** Either assert `target.shape == (nphi, ntheta)` (or accept
shape `()` / a scalar) at the kernel entry, or push the shape validation
into `make_fixed_surface_flux_spec` (`specs.py:1397`) where the surface
contract is materialized.

### HIGH-2 - `"local"` gradient is silently finite at `|B|=0` (mixed quadrature)

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:65-78` (kernel)
**File:** `src/simsopt/objectives/fluxobjective_jax.py:54-63` (adapter guard)

When exactly one quadrature point has `|B|=0` and other points are finite,
the JAX kernel returns `J = +inf` (correct), but `jax.grad` returns a
**finite vector** for the surviving points. The singular-point gradient is
`[0, 0, 0]` (because `jnp.where(singular, ..., ...)` zeros out its branch),
not `nan`. Concretely:

```
nphi=2, ntheta=2
B[0,0] = 0   # singular
B[0,1] = e_x, B[1,0] = 0.5 e_x, B[1,1] = 0.3 e_x
normal[i,j] = (1,1,1)/sqrt(3),  target = 0
J_local = +inf
grad B = [
  0,          0,           0,            # B[0,0] (singular)
  -7.4e-19, 0.0833,      0.0833,         # B[0,1]
  -1.5e-18, 0.1667,      0.1667,         # B[1,0]
   2.8e-17, 0.2778,      0.2778,         # B[1,1]
]
```

The adapter's `_raise_if_nonfinite_squared_flux_gradient`
(`fluxobjective_jax.py:54-63`) inspects `np.isfinite(value)` **and**
`np.isfinite(grad)`. Because `value` is `+inf`, the guard does fire and the
adapter raises `ObjectiveFailure`. So far so good for the **`SquaredFluxJAX`
adapter**.

However:

1.  Callers of the **bare kernel** (`integral_BdotN` directly, or the
    JAX-only `coil_current_fixed_geometry_value_and_grad_jax` helper at
    `fluxobjective_jax.py:123`) get back a finite-looking gradient even
    though they are at a singular configuration. The CPU reference
    `SquaredFlux.dJ()` raises `ObjectiveFailure` *with an explicit message*
    citing `|B|^2 vanishes` (`fluxobjective.py:101-106`), so a downstream
    `except ObjectiveFailure` clause that previously caught only the C++
    semantics will *not* fire on the bare JAX path.

2.  The adapter relies on `value == inf` being available *together* with
    `grad`. If a future refactor caches `grad` independently of `value`
    (e.g., a JIT-traced grad-only path that bypasses the value), the finite
    grad would slip through. There is no defense-in-depth.

Reproduction (probe used in audit):
```
.conda/jax/bin/python -c "
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from simsopt.objectives.integral_bdotn_jax import integral_BdotN
B = jnp.array([[[0,0,0],[1,0,0]],[[0.5,0,0],[0.3,0,0]]], dtype=jnp.float64)
n = jnp.ones((2,2,3))/jnp.sqrt(3); t = jnp.zeros((2,2))
print(float(integral_BdotN(B, t, n, 'local')))            # inf
print(jax.grad(lambda B: integral_BdotN(B,t,n,'local'))(B))  # finite!
"
```

**Severity rationale:** The CPU's `SquaredFlux.dJ` raises a *typed* error;
the JAX kernel returns a finite gradient. The adapter intercepts via the
`value=inf` check, but only because `value` is co-emitted, and the error
message is generic ("singular because the objective or its derivative is
non-finite") rather than the C++'s specific physics-grounded message. Any
downstream code that depends on the **error type** or **message** to
diagnose the failure mode will see different behavior depending on whether
it ran against CPU or JAX. **HIGH.**

**Fix sketch:** make the kernel emit `nan` (not `0.0`) for the
singular-point local gradient (by routing through `jnp.where(singular,
jnp.nan, ...)` on the *post-grad* path, or via a custom_vjp that returns
`nan` for the gradient when `singular`). Alternatively, the adapter could
explicitly check `singular = (B2 <= 0) & has_normal` before calling JAX
and raise a typed error that matches the CPU's message.

### HIGH-3 - `_raise_if_surface_dofs_drifted` has zero test coverage

**File:** `src/simsopt/objectives/fluxobjective_jax.py:378-386`

The fingerprint guard is well-designed (it hashes `surface.local_full_x`
via blake2b-128 and so even fully-fixed surfaces' silent mutation via
`set_rc` will be caught - verified empirically below) and the docstring at
lines 87-101 explicitly justifies the choice of `local_full_x` over `x`.
But the guard is exercised only in the audit's manual probe.

Search of the test corpus:
```
grep -rn 'captures fixed surface geometry at construction' tests/  -> NO HITS
grep -rn 'raise_if_surface_dofs_drifted'                  tests/  -> NO HITS
grep -rn 'surface_dofs_fingerprint'                       tests/  -> NO HITS
```

In contrast:
- `_raise_if_field_points_drifted` has a dedicated test:
  `tests/integration/test_stage2_jax.py:2207-2226`
  (`test_j_rejects_mutated_field_points_after_construction`).
- `_raise_if_field_dof_layout_drifted` has no direct test but is at least
  exercised by the layout-mutating path indirectly.

Empirical confirmation (audit probe) that the guard works:
```
surface.fix_all()                            # all DOFs fixed before construction
jf = SquaredFluxJAX(surface, bs_jax)
J1 = jf.J()                                  # 7.0e-5
surface.set_rc(1, 0, 0.25)                   # mutate FIXED DOF
jf.J()                                       # raises RuntimeError
```

So the code works; the *test* is missing. Until a regression test pins it,
the guard could be silently deleted in a code-simplification pass and no CI
job would notice. Worth noting: the docstring explicitly motivates the
choice of `local_full_x` over `x` ("a surface that is fully fixed at
construction would otherwise hash an empty array"), so the lack of a
regression test for that exact scenario is especially load-bearing.

**Severity rationale:** the guard prevents silent stale-cache wrong-answer
bugs; losing it without warning would land us in exactly the class of
parity-failure bug the audit is supposed to prevent. **HIGH.**

**Fix sketch:** add three tests:
1. `test_j_rejects_surface_set_rc_after_construction` (use `set_rc` to mutate a
   free DOF; verify `J()` raises).
2. `test_j_rejects_fixed_surface_dof_mutation_after_fix_all` (use `fix_all()`
   then `set_rc`; verify guard still fires because fingerprint covers
   `local_full_x`).
3. `test_j_rejects_surface_x_setter_after_construction` (use `surface.x =
   surface.x + 0.01` to bypass `set_rc` and exercise the alternate writer).

---

### MEDIUM-1 - Empty-mesh parity divergence (`nphi=0`, `ntheta=0`)

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:125` (`J = 0.5 *
scalar_square_sum(residual, ...)`)
**File:** `src/simsoptpp/integral_BdotN.cpp:119` (`result = 0.5 * numerator_sum / (nphi * ntheta)`)

For `B.shape == (0, 0, 3)`:
- CPU returns `nan` (because `numerator_sum / (nphi*ntheta) = 0/0`).
- JAX returns `0.0` (because `pairwise_sum_flat` on a size-0 input returns
  `jnp.sum(empty) = 0`).

Reproduction (audit probe):
```
B = np.zeros((0,0,3)); t = np.zeros((0,0)); n = np.zeros((0,0,3))
sopp.integral_BdotN(B, t, n, "quadratic flux")  # nan
integral_BdotN(jnp.asarray(B), jnp.asarray(t), jnp.asarray(n), "quadratic flux")  # 0.0
```

Same divergence at `nphi=0, ntheta=4` (CPU `nan`, JAX `0.0`).

**Severity:** Unlikely to bite production (`nphi=0` surfaces are nonsense),
but it violates the parity-ladder claim of "byte identity to the C++
oracle" for all valid kernel inputs. MEDIUM because it's a measurable
divergence with a definite cause, but doesn't affect any current call site.

### MEDIUM-2 - `target=None` path NaN-propagates from `normal`

**File:** `src/simsopt/jax_core/objectives_flux.py:38-42`

```python
def _fixed_surface_target_array(normal, target):
    if target is None:
        zero_target = jnp.sum(normal, axis=-1)
        return zero_target - zero_target
    return _as_jax_float64(target)
```

The `zero_target - zero_target` idiom (chosen presumably to avoid
host-allocating a fresh `jnp.zeros((nphi,ntheta))` and tripping a transfer
guard) propagates NaN/Inf in `normal`: `nan - nan = nan`. The CPU
`SquaredFlux.__init__` (`fluxobjective.py:71`) does
`np.zeros(self.surface.normal().shape[:2])`, which is *unconditional* zero
even if normals are nonsensical.

Audit probe:
```
normal = jnp.array([[[1.0, 2.0, jnp.nan]]])
_fixed_surface_target_array(normal, None)  # array([[nan]])
```

**Severity:** Diagnostic poison - a `target=None` Stage 2 solve with even
a single bad normal component will get a NaN-poisoned baseline target. CPU
would have a clean zero target and the failure would surface later (in
gamma/normal evaluation). The first place to NaN flips downstream. MEDIUM.

**Fix sketch:** allocate `jnp.zeros(normal.shape[:2], dtype=jnp.float64)`
explicitly, or `jnp.zeros_like(normal[..., 0])`.

### MEDIUM-3 - Test fixture nphi/ntheta is below production-floor

**Files:** `tests/objectives/test_integral_bdotn_jax.py:44`,
`tests/objectives/test_integral_bdotn_item10_closeout.py:48-51`

Default test fixture: `nphi=10, ntheta=12` (`_make_test_data`). C++ parity
test (`test_cpp_parity`): `nphi=15, ntheta=15`. Closeout chained test:
`nphi=16, ntheta=8` (the *production floor*).

Production single-stage solves typically use `nphi=64, ntheta=64`
(`tests/test_lightning_production_gpu_proof.py:308`,
`tests/test_runpod_single_stage_continuation.py:278-733`). Reduction-order
drift between `pairwise_sum_flat`(JAX) and OpenMP `reduction(+:)`(C++)
is `O(eps * sqrt(N))` for tree-reduce vs `O(eps * N)` for sequential
summation, so the small fixtures hide accumulation drift that would
become visible at 4096-point grids. The closeout test is at 128 points
(`16 * 8`); the production floor is `64*64 = 4096`. No same-state
byte-identity gate at 4096 points exists for `"normalized"` (the lane
where reduction-order matters most).

**Severity:** the parity contract advertises byte identity at production
floors but the regression tests don't exercise that floor for the
`integral_BdotN` chain. MEDIUM (test coverage gap, not a math bug).

### MEDIUM-4 - `signed_BdotN_flux` is exposed publicly with no production caller

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:34, 86`

`__all__ = ["integral_BdotN", "residual_BdotN", "signed_BdotN_flux"]`.

But `grep -rn signed_BdotN_flux src/ benchmarks/ examples/` returns **only
the definition site** and the single test that calls it
(`tests/objectives/test_integral_bdotn_jax.py:152`). No production code
consumes it. It's effectively dead public API.

Worse, broadcasting silently works:
```
B = jnp.ones((2, 3, 3))
normal_broadcasted = jnp.ones((1, 3, 3))
signed_BdotN_flux(B, normal_broadcasted)  # = 3.0, no error
```

so a stale caller that accidentally passes a mis-shaped normal will get a
finite number instead of a shape error.

**Severity:** API surface clutter / silent-shape-mismatch hazard. MEDIUM
(prune or guard).

---

### LOW-1 - Silent float32 acceptance vs float64 contract

**Files:** `src/simsopt/objectives/integral_bdotn_jax.py:42-89`,
`src/simsopt/jax_core/_math_utils.py:40-41`

`as_jax_float64` (used inside `FixedSurfaceFluxSpec` construction at
`specs.py:1404-1408`) coerces inputs to float64. But the *raw kernel*
`integral_BdotN(B, target, normal, ...)` does **not** apply
`as_jax_float64` to its inputs. If the caller passes `B` as float32, the
kernel runs at float32 throughout, returns a float32 scalar, and the
host-side `host_scalar(...)` cast to float64 papers over the precision
loss too late.

Audit probe:
```
B32 = np.ones((4,4,3), dtype=np.float32); B32[..., 2] = 1.0
target64 = np.zeros((4,4), dtype=np.float64)
normal64 = np.zeros((4,4,3), dtype=np.float64); normal64[..., 2] = 1.0
integral_BdotN(jnp.asarray(B32), jnp.asarray(target64), jnp.asarray(normal64), "quadratic flux")
# -> dtype=float64 (promoted)  via numpy promotion rules

integral_BdotN(jnp.asarray(B32), jnp.asarray(target64.astype(np.float32)),
               jnp.asarray(normal64.astype(np.float32)), "quadratic flux")
# -> dtype=float32   <-- silent precision loss
```

The CPU C++ also accepts float32 (xtensor-python implicit cast), so this is
parity-compatible in the *value* sense, but the JAX path doesn't even cast
to float64 on the way in. Production callers go through
`as_jax_float64`-guarded specs, but the bare kernel is part of the public
`__all__`.

**Severity:** LOW (the bare kernel is rarely called outside specs path; CPU
is also lax). Worth a runtime promotion / assertion if the kernel is
formally part of the public API.

### LOW-2 - Complex inputs silently produce |r|^2 instead of an error

**Files:** `src/simsopt/objectives/integral_bdotn_jax.py:42-83`,
`src/simsopt/jax_core/reductions.py:113-120`

```
scalar_square_sum (reductions.py:113-115):
  flat = jnp.ravel(jnp.asarray(array))
  squared = (jnp.conj(flat) * flat).real
```

This is the **Hermitian** square sum, not the analytic-complex square
sum (`r * r`, no conjugate). A user passing complex `B` accidentally gets
the magnitude-squared (`|B|^2`) rather than an error. The C++ path can't
even accept complex inputs (xtensor-python `pyarray<double>` only).

Audit probe:
```
B_c = jnp.array([[[1.0+2.0j, 0, 0]]], dtype=jnp.complex128)
normal = jnp.array([[[1.0, 0, 0]]], dtype=jnp.float64)
target = jnp.array([[0.0]], dtype=jnp.float64)
integral_BdotN(B_c, target, normal, "quadratic flux")
  -> 2.5  (= 0.5 * |1+2j|^2 = 0.5 * 5)
```

**Severity:** LOW. Unlikely accidental input. The Hermitian sum is at least
mathematically meaningful (it's the squared modulus), so it's not "garbage
in, garbage out", just "complex in, magnitude out without an error".

### LOW-3 - Error type for unknown definition diverges at kernel level

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:80`
(`raise ValueError(f"Unknown definition: {definition!r}")`)
**Counterpart:** `src/simsoptpp/integral_BdotN.cpp:57`
(`throw std::runtime_error("Unrecognized value for 'definition'.")`)

JAX raises `ValueError`; pybind11 surfaces the C++ `std::runtime_error` as
`RuntimeError`. The two adapter classes both raise `ValueError` at the
adapter boundary, so callers that wrap `SquaredFlux(...)` see consistent
typed errors. But callers using the *bare kernel*
(`sopp.integral_BdotN(...)` vs `integral_bdotn_jax.integral_BdotN(...)`)
need different `except` clauses. Production hot path is the adapter, so
this is LOW (kernel-level only).

### LOW-4 - Asymmetric algebraic form for `"normalized"` (defense in depth)

**File:** `src/simsopt/objectives/integral_bdotn_jax.py:55-64`

Documented as L-1 in pass-1; reiterated here because the audit also
confirmed that no same-state byte-identity test exists for `"normalized"`
at `nphi*ntheta >= 64*64`. The `pairwise_sum_flat(B2 * norm_n)` (line 57)
is the JAX denominator, but the asymmetric form
`residual = BdotN * jnp.sqrt(weight)` followed by squared accumulation
makes byte-identity with `0.5 * numerator / denominator` (C++) implausible
at production scale. The `_normalized_reduction_stress_data` test
(`test_integral_bdotn_jax.py:244`) only verifies that `J=0.5` to acceptance
tolerance, not byte identity, so we don't *measure* the divergence either.

**Severity:** LOW (already documented as MEDIUM in pass-1; reiterating that
the test for actual byte-identity drift at production scale is missing).

### LOW-5 - `field.set_points()` no-op guard relies on internal version counter

**File:** `src/simsopt/objectives/fluxobjective_jax.py:225-226, 358-365`

The drift detection uses `field._points_version`, a private internal counter
on `BiotSavartJAX`. The contract is: "rebuild the objective for a new point
set". But there is no test that `field.set_points(field.get_points())`
(re-setting the same points) does NOT trip the guard. If a caller does this
defensively, the audit can't tell from the code whether the version counter
treats the no-op as a bump.

**Severity:** LOW (test gap; not directly a parity bug).

### LOW-6 - JIT cache key signature is implicit / undocumented

**File:** `src/simsopt/objectives/fluxobjective_jax.py:250-261`

`_bind_native_forward` binds `jit_forward` and `jit_val_grad` at construction
time but does not document what the JIT cache key includes: `flat_dofs`
shape/dtype, `flux_spec` pytree shape, and the closed-over `surface` /
`coil` quantities. If a future change adds an option that's not part of the
spec but affects forward semantics, the JIT will silently reuse the cached
program against new semantics.

**Severity:** LOW (future-bug-shaped, not a current bug).

---

### INFO-1 - C++ side effects audit: none

Re-read `integral_BdotN.cpp:12-123` end-to-end against UB / mutation
hazards:

- No `static` state.
- `mod_B_squared` is now scoped *inside* the loop body (line 65), so the
  earlier race condition (flagged and fixed per `CLAUDE.md`) is verified
  fixed in this commit.
- OpenMP reductions are well-formed: `reduction(+:numerator_sum,
  denominator_sum) reduction(max:has_local_singularity)`. `has_local_singularity`
  is `int 0/1` and `max` is correct for "any singular point exists".
- No raw-pointer aliasing: `Bcoil_ptr`, `Btarget_ptr`, `n_ptr` are all
  read-only, no writes. xtensor-python ensures distinct allocations.
- No signed/unsigned mixing: indexed loop is `int i` against
  `nphi*ntheta`, both `int` from xtensor shape.
- No throw inside the parallel region except the `should never reach this
  point` branch (line 105) which is unreachable - `definition_int` is
  always one of three known values.
- No allocation inside the loop; per-thread state is stack-local doubles.

**Conclusion:** C++ side has no detected UB / race / leak in the current
commit. The `mod_B_squared` race that `CLAUDE.md` flagged as
"previously-fixed" is confirmed still fixed.

### INFO-2 - Sign-convention is consistent (CPU vs JAX, all three defs)

All three definitions and both implementations use `r = B.n_hat - target`:
- C++: `BcoildotN -= Btarget_ptr[i]` (line 85).
- JAX: `BdotN = jnp.sum(Bcoil * unit_n, axis=-1) - target` (line 50).
- CPU SquaredFlux: `B_n = Bcoil_n - self.target` (`fluxobjective.py:92`).

All four squared-residual branches use `r^2` (sign-invariant). No
sign-convention drift detected.

### INFO-3 - `target=None` -> empty array (CPU) vs allocated zeros (JAX adapter)

CPU `SquaredFlux.__init__` (`fluxobjective.py:69-71`) allocates a
`np.zeros((nphi, ntheta))` and stores it. CPU C++ separately accepts an
empty `Btarget` (size 0) as "no target", with `Btarget_ptr=NULL`. JAX
adapter (`fluxobjective_jax.py:214-219`) passes `target_array=None` through
to `make_fixed_surface_flux_spec`, which calls `_fixed_surface_target_array`
that returns `jnp.sum(normal, axis=-1) - jnp.sum(normal, axis=-1)`. Result
is mathematically identical for clean inputs but differs in NaN propagation
(see MEDIUM-2).

Note that `SquaredFlux.J()` does *not* take advantage of the C++ `Btarget`
size-0 fast path; it always passes the allocated target array. So this
divergence is only relevant for direct kernel callers of
`sopp.integral_BdotN` with `target=np.array([])`.

---

## Untested edge-case inventory (concrete inputs)

| # | Input | Definition | Expected | Actual (JAX) | Actual (CPU) | Currently tested? |
|---|-------|------------|----------|--------------|--------------|-------------------|
| 1 | `target.shape=(5,7)`, `B.shape=(1,1,3)` | quadratic flux | error | `17.5` (silent) | RuntimeError | NO |
| 2 | `target.shape=(3,2)`, `B.shape=(2,3,3)` | quadratic flux | error | TypeError (broadcasting) | RuntimeError | NO |
| 3 | scalar `target=0.0`, B/n nonzero | quadratic flux | accept (uniform offset) | accepted | accepted | NO |
| 4 | mixed `\|B\|=0` at one point, finite elsewhere | local | grad raises typed err | grad finite (silent), `J=inf` triggers adapter `ObjectiveFailure` | `dJ` raises `ObjectiveFailure` with explicit `\|B\|^2` message | NO (bare kernel) |
| 5 | `target == B.n_hat` exactly | quadratic flux | `J=0` and grad=0 | OK | OK | NO |
| 6 | `nphi=1, ntheta=1`, valid B/n | quadratic flux | finite | OK | OK | NO |
| 7 | `nphi=0, ntheta=0` | quadratic flux | undefined | `0.0` | `nan` | NO |
| 8 | `nphi=0, ntheta>0` | any | undefined | `0.0` | `nan` | NO |
| 9 | `target=None` and `normal` contains NaN | quadratic flux | match CPU | NaN-poisoned target | CPU target is zeros (clean) | NO |
| 10 | float32 inputs throughout | any | upcast to float64 or error | float32 output (silent precision) | float32 -> float64 upcast (xtensor) | NO |
| 11 | complex128 `B` | any | error | `\|B\|^2` interpretation (silent) | error (pybind dtype mismatch) | NO |
| 12 | integer `B` | any | upcast or error | upcast to float64 (silent) | error (pybind dtype) | NO |
| 13 | unknown `definition="Quadratic Flux"` | - | error | ValueError | RuntimeError | partial (only "invalid") |
| 14 | empty `definition=""` | - | error | ValueError | RuntimeError | NO |
| 15 | surface DOF mutated via `set_rc` after construction | adapter | RuntimeError | RuntimeError (verified by audit, not by test) | n/a | NO |
| 16 | surface fully fixed then DOF mutated | adapter | RuntimeError | RuntimeError (verified by audit) | n/a | NO |
| 17 | `field.set_points(field.get_points())` (no-op) | adapter | no error | unknown | n/a | NO |
| 18 | Axisymmetric surface where normalized flux is identically zero by symmetry | normalized | well-defined (`Sigma B^2 \|n\| > 0`, J = 0) | (untested) | (untested) | NO |
| 19 | `B = 0` everywhere on surface with positive area | normalized | `+inf` | OK (verified by `test_zero_field_normalized_returns_inf`) | OK | YES |
| 20 | nphi=64, ntheta=64, normalized definition, same-state byte identity | normalized | byte identity to C++ | (untested) | (untested) | NO |

---

## Recommended new test fixtures

### A. Kernel-level shape validation tests (closes HIGH-1)

```python
# tests/objectives/test_integral_bdotn_jax.py - new class
class TestIntegralBdotNShapeValidation:
    @pytest.mark.parametrize("definition", _DEFINITIONS)
    def test_wrong_target_shape_raises(self, definition):
        B = jnp.ones((1, 1, 3))
        normal = jnp.ones((1, 1, 3))
        target_wrong = jnp.zeros((5, 7))  # broadcasts silently today
        with pytest.raises((ValueError, AssertionError),
                           match="target.*shape"):
            integral_BdotN(B, target_wrong, normal, definition)

    def test_transposed_target_raises(self):
        B = jnp.ones((2, 3, 3))
        normal = jnp.ones((2, 3, 3))
        target = jnp.zeros((3, 2))   # right total, wrong shape
        with pytest.raises((TypeError, ValueError, AssertionError)):
            integral_BdotN(B, target, normal, "quadratic flux")
```

### B. Local-singularity gradient tests (closes HIGH-2)

```python
def test_local_grad_at_singular_point_marks_failure_explicitly():
    """A single |B|=0 quadrature point must surface a typed error, not a finite grad."""
    nphi, ntheta = 2, 2
    B = jnp.array([[[0.0,0.0,0.0], [1.0,0,0]],
                   [[0.5,0,0], [0.3,0,0]]], dtype=jnp.float64)
    n = jnp.ones((nphi, ntheta, 3), dtype=jnp.float64) / jnp.sqrt(3)
    t = jnp.zeros((nphi, ntheta), dtype=jnp.float64)
    val = float(integral_BdotN(B, t, n, "local"))
    assert np.isinf(val)
    grad = jax.grad(lambda b: integral_BdotN(b, t, n, "local"))(B)
    # Today: grad is finite (silent). Should be nan/inf so adapters can detect.
    assert not np.all(np.isfinite(grad)), \
        "JAX 'local' gradient must be non-finite at |B|=0 quadrature points"
```

If the gradient pathway is intentionally kept silent (per the adapter's
guard), at minimum add an adapter-level test that the typed
`ObjectiveFailure` is raised with the same message as CPU `SquaredFlux.dJ`:

```python
def test_squared_flux_jax_local_singular_raises_with_consistent_message():
    # Build a surface + coils where |B|=0 at one quadrature point
    # (e.g., place an evaluation point at the magnetic null of a single coil)
    jf = SquaredFluxJAX(surf, bs_jax, definition="local")
    with pytest.raises(ObjectiveFailure, match=r"\|B\|.*vanish"):  # CPU msg
        jf.dJ()
```

### C. Surface-drift guard tests (closes HIGH-3)

```python
def test_j_rejects_surface_set_rc_after_construction(jax_stage2_setup):
    surf, bs_jax = jax_stage2_setup
    jf = SquaredFluxJAX(surf, bs_jax)
    jf.J()
    surf.set_rc(1, 0, surf.get_rc(1, 0) + 0.01)
    with pytest.raises(RuntimeError, match="captures fixed surface geometry"):
        jf.J()
    with pytest.raises(RuntimeError, match="captures fixed surface geometry"):
        jf.dJ()

def test_j_rejects_fully_fixed_surface_dof_mutation(jax_stage2_setup):
    surf, bs_jax = jax_stage2_setup
    surf.fix_all()                              # before constructing objective
    jf = SquaredFluxJAX(surf, bs_jax)
    surf.set_rc(1, 0, surf.get_rc(1, 0) + 0.01) # mutates fixed DOF
    with pytest.raises(RuntimeError, match="captures fixed surface geometry"):
        jf.J()
```

### D. Empty/single-point boundary parity (closes MEDIUM-1)

```python
@pytest.mark.parametrize("definition", _DEFINITIONS)
def test_empty_mesh_matches_cpp(self, definition):
    """0-point quadrature must NOT silently return 0 if CPU returns NaN."""
    B = np.zeros((0, 0, 3))
    target = np.zeros((0, 0))
    normal = np.zeros((0, 0, 3))
    J_cpu = sopp.integral_BdotN(B, target, normal, definition)
    J_jax = float(integral_BdotN(jnp.asarray(B), jnp.asarray(target),
                                  jnp.asarray(normal), definition))
    if np.isnan(J_cpu):
        assert np.isnan(J_jax) or pytest.raises(...)  # either match or error
    else:
        np.testing.assert_allclose(J_jax, J_cpu, rtol=1e-13)

@pytest.mark.parametrize("definition", _DEFINITIONS)
def test_single_point_grid_matches_cpp(self, definition):
    B = np.array([[[0.1, 0.0, 1.0]]]); target = np.zeros((1, 1))
    normal = np.zeros((1, 1, 3)); normal[..., 2] = 1.0
    J_cpu = sopp.integral_BdotN(B, target, normal, definition)
    J_jax = float(integral_BdotN(jnp.asarray(B), jnp.asarray(target),
                                  jnp.asarray(normal), definition))
    np.testing.assert_allclose(J_jax, J_cpu, rtol=1e-13)
```

### E. Production-scale (64x64) reduction-order drift gate (closes MEDIUM-3 + LOW-4)

```python
@pytest.mark.parametrize("definition", _DEFINITIONS)
def test_production_scale_64x64_cpp_parity(self, definition):
    """At production scale, byte-identity to C++ must still hold per direct_kernel."""
    B, target, normal = _make_test_data(nphi=64, ntheta=64, seed=11)
    J_cpu = sopp.integral_BdotN(np.asarray(B), np.asarray(target), np.asarray(normal),
                                 definition)
    J_jax = float(integral_BdotN(B, target, normal, definition))
    np.testing.assert_allclose(J_jax, J_cpu, rtol=_DIRECT_KERNEL["rtol"],
                                              atol=_DIRECT_KERNEL["atol"])
```

### F. Dtype contract tests (closes LOW-1, LOW-2)

```python
def test_float32_input_raises_or_upcasts():
    B32 = jnp.ones((4, 4, 3), dtype=jnp.float32)
    target = jnp.zeros((4, 4), dtype=jnp.float32)
    normal = jnp.zeros((4, 4, 3), dtype=jnp.float32); normal = normal.at[..., 2].set(1.0)
    # Either raise OR promote to float64. The current silent float32 result is
    # not a defensible contract.
    out = integral_BdotN(B32, target, normal, "quadratic flux")
    assert out.dtype == jnp.float64, \
        f"integral_BdotN must keep float64 contract; got dtype={out.dtype}"

def test_complex_input_raises():
    B_c = jnp.array([[[1.0+2.0j, 0, 0]]], dtype=jnp.complex128)
    normal = jnp.array([[[1.0, 0, 0]]], dtype=jnp.float64)
    target = jnp.array([[0.0]], dtype=jnp.float64)
    with pytest.raises((TypeError, ValueError), match="complex"):
        integral_BdotN(B_c, target, normal, "quadratic flux")
```

### G. Target=None with NaN-tainted normal (closes MEDIUM-2)

```python
def test_target_none_does_not_propagate_normal_nan():
    """Internal zero-target allocation must not propagate NaN from the normal."""
    normal = jnp.array([[[1.0, 2.0, jnp.nan]]])
    target = _fixed_surface_target_array(normal, None)
    assert not np.any(np.isnan(target)), \
        "target=None path must allocate a clean zero array"
```

### H. Drop or guard `signed_BdotN_flux` (closes MEDIUM-4)

Two options:
1. Remove from `__all__` (it's still importable, just not advertised).
2. Add shape validation + a per-point oracle test against a hand-computed
   reference (not just the closed-curve invariant).

---

## Summary

Pass-1 verified forward-formula parity at `direct_kernel` tolerance and
flagged the `"normalized"` reduction-order divergence (M-1) plus the
JAX-only public symbols (I-1).

This deeper pass added three HIGH findings (silent target broadcasting,
silent `local` singular-point gradient, untested surface-drift guard),
four MEDIUMs (empty-mesh parity divergence, `target=None` NaN poison,
production-scale test gap, dead-API exposure), and six LOWs (dtype,
complex, error-type kernel divergence, defense-in-depth for normalized
byte-identity, set-points no-op, JIT cache key documentation). The C++
side is clean for UB / OpenMP / aliasing; the `mod_B_squared` race
flagged in `CLAUDE.md` is verified fixed.

All findings are accompanied by concrete reproduction probes executed
against the live `.conda/jax` (`jax==0.10.0`) environment. None of the
findings are theoretical: every divergence has an input that triggers it
and an output that demonstrates the gap. Recommended new test fixtures
above close all eight novel gaps; pinning them in CI would convert the
"audit found this" status into a regression-protected contract.
