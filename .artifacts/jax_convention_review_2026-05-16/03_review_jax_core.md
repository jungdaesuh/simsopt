# `src/simsopt/jax_core/` audit — convention + JAX best-practice review

Reviewer: Opus 4.7 (max-effort lane), 2026-05-16
Worktree: `/Users/suhjungdae/code/columbia/simsopt-jax`
Branch: `gpu-purity-stage2-20260405`
Runtime: JAX 0.10.0 / jaxlib 0.10.0 / Python 3.11 / NumPy 2.x

Scope: every file under `src/simsopt/jax_core/` (37 files, ~22K LOC). Tests
and the adapter layer in `field/*_jax*.py` / `geo/*_jax*.py` are explicitly
out of scope.

## Executive summary

The `jax_core/` package is, in the main, a clean and well-factored pure-JAX
kernel layer. The recurring themes worth surfacing for triage are:

1. **BLOCKER (CONVENTION):** `jax_core/` is **not** in fact dependency-free
   from the wider `simsopt.geo` / `simsopt.field` / `simsopt.objectives`
   layers. There are 9 cross-package imports — including some at module top
   level — that transitively pull in `Optimizable`, the `_simsoptpp` shim,
   and adapter wrappers. This violates the CLAUDE.md "no simsoptpp
   dependency" rule for the kernel layer, blocks publishing this package as
   a JAX-only library, and is partially confirmed by the
   `project_curve_jax_core_import_cycle.md` memory note. See findings
   §C-01 / §C-02.

2. **HIGH (BEST-PRACTICE):** `magnetic_axis_helpers.on_axis_iota_rk` and
   the five `trace_*` drivers in `tracing.py` all wrap their integration
   loops in `jax.lax.while_loop`. `while_loop` is **not reverse-mode
   differentiable** in JAX. The docstring on `on_axis_iota_rk` advertises
   gradient support ("differentiable through field DOFs as long as the
   user supplies a JAX-traceable field-evaluation callback") which is
   misleading: `jax.grad(on_axis_iota_rk)` will raise. See §B-01.

3. **HIGH (BEST-PRACTICE / CONVENTION):** `analytic_pure_fields.py`
   declares — and the wrappers `toroidal_dB`, `toroidal_dA`,
   `poloidal_dB` honour — a `dB[p, l, j]` axis layout where axis 1 is the
   `B` component and axis 2 is the derivative direction. `mirror_dB`
   in the same file uses the documented CLAUDE.md convention
   (`dB[p, j, l] = ∂_j B_l`). This split (literally documented as
   "intentional" so direct-kernel parity is bit-tight against the CPU
   oracle) means a downstream consumer who reads CLAUDE.md and assumes
   one mental model gets the wrong axes from half of these kernels.
   See §B-02.

4. **HIGH (CONVENTION/IMMUTABILITY):**
   `InterpolatedBoozerFieldFrozenState.specs` is a mutable `dict`
   wrapped inside a `frozen=True` dataclass and is
   "**mutated in place** by `InterpolatedBoozerFieldJAX._ensure_spec`
   when a scalar is lazy-built". This is an explicit, documented
   IMMUTABLE-rule violation. Once a `state` reference is captured in a
   JIT trace or pytree compare, the silent mutation will surface as
   cache-coherency surprises (specifically, the pytree leaf set is
   unstable). See §B-03.

5. **HIGH (PERF/CORRECTNESS):**
   `magnetic_axis_helpers.on_axis_iota_rk` finishes with `jnp.linalg.eig`
   on a 2x2 matrix (line 598). In jaxlib 0.10.0 `eig` is implemented via
   a LAPACK host-callback for non-Hermitian inputs; on GPU this **forces
   a device-to-host round trip per call**, defeating the same
   `transfer_guard("disallow")` discipline the rest of the file (and
   `_device_scalars.py`, `_math_utils.py`) goes to great lengths to
   maintain. See §B-04.

6. **HIGH (PERF/COMPILE):** The kernel-factory in
   `biotsavart.py::_make_kernel` is keyed on
   `(integrand_key, diff_mode, coil_cs, quad_bs, point_cs,
   point_vma_axis_name, jax.default_backend())`. The platform string is
   bound at call time, so a host process that toggles
   `SIMSOPT_JAX_PLATFORM` between calls will compile two distinct kernel
   sets but **share the `lru_cache(maxsize=32)`** with the same outer
   `_make_kernel`. The closure body explicitly does `del platform`, so
   the two compiled kernels are functionally identical to XLA but
   represent independent Python objects. This is correct, just wasteful.
   See §B-05.

7. **MEDIUM (CORRECTNESS):** `biotsavart.py::_safe_radius_squared`
   clamps `r²` to `1e-60`, an intentional **divergence from the C++
   oracle's NaN/Inf-on-coil behaviour** documented in the source. Two
   downstream implications: (a) `dB/dX` at point-on-coil will silently
   return a finite but garbage tensor; (b) callers cannot detect
   point-on-coil via finiteness. Worth a one-line warning in the
   `biot_savart_*` public docstrings, not just the private helper.
   See §B-06.

8. **MEDIUM (CONVENTION):** Several specs use `meta_fields` to store
   data that *might* legitimately be optimized over (`CurveHelicalSpec`
   has `R0`/`r` in meta; `circular_coil.CircularCoilSpec` puts `r0`,
   `Inorm`, `center`, `normal` in meta). This is fine when those
   parameters are fixed but means **autodiff through those parameters
   is impossible without re-tracing**. The current set of consumers
   evidently treats them as fixed; downstream contributors should be
   warned. See §B-07.

9. **MEDIUM (BEST-PRACTICE):** `scalar_potential_rz.py` adds
   `1e-30 * phi * R * Z` to every output expression (line 18 +
   line 35-62) as a hack to defeat SymPy's constant-folding to a
   numeric zero. This injects a systematic, deterministic bias to every
   B/dB output. At float64 the bias is below the noise floor on
   realistic geometries; nonetheless this is a documented "ghost term"
   that will surface in convergence-grade unit tests. See §B-08.

10. **LOW (BEST-PRACTICE):** Naming inconsistency between `jnp.atan2`
    and `jnp.arctan2` across files. Most of `jax_core/` uses
    `jnp.arctan2` (NumPy-compatible). `analytic_pure_fields.py` and
    `dipole_field.py` use `jnp.atan2` (Python `math` style). Both are
    available in JAX 0.10.0; the inconsistency is a style smell only.
    See §B-09.

The remaining findings are NITs that do not affect correctness or
performance but document conventions for future contributors.

---

## A. simsopt-convention compliance

### §A-01. No direct `simsoptpp` import — confirmed pass

Searched `jax_core/` for `import simsoptpp` and `from simsopt._core`:
zero hits at any depth. Good.

### §A-02. Layout vs C++ counterparts — confirmed pass

Every JAX module has a clearly identified C++ peer documented in its
module docstring or commit history. The mapping is consistent.

- `jax_core/biotsavart.py` ↔ `simsoptpp/biot_savart_impl.h`
- `jax_core/wireframe.py` ↔ `simsoptpp/wireframe_field_impl.h`,
  `magneticfield_wireframe.cpp`
- `jax_core/regular_grid_interp.py` ↔
  `simsoptpp/regular_grid_interpolant_3d.h` /
  `regular_grid_interpolant_3d_impl.h`
- `jax_core/dipole_field.py` ↔ `simsoptpp/dipole_field.cpp` / `.h`
- `jax_core/tracing.py` ↔ `simsoptpp/tracing.cpp`
- `jax_core/analytic_fields.py` ↔ `simsoptpp/dommaschk.cpp`,
  `simsoptpp/reiman.cpp`
- `jax_core/boozer_radial_interp.py` ↔
  `simsoptpp/boozerradialinterpolant.cpp`
- `jax_core/pm_optimization.py` ↔
  `simsoptpp/permanent_magnet_optimization.cpp`
- `jax_core/interpolated_boozer_field.py` ↔
  `simsoptpp/boozermagneticfield_interpolated.h`

### §A-03. Stellsym DOF convention — confirmed pass

`specs.py:1572` imports `stellsym_scatter_indices` from
`..geo.surface_fourier_jax` (lazily, inside `make_*_spec`) which is the
SSOT for the cos-cos+sin-sin (x) / cos-sin+sin-cos (y, z) ordering. No
`jax_core/` module re-implements the ordering. (The import cycle this
creates is recorded as a separate finding — see §C-01.)

### §A-04. `dB/dX[p, j, l] = ∂_j B_l` axis convention — partial pass

The convention is the documented SIMSOPT default, repeated in
`CLAUDE.md`. `jax_core/` modules **mostly** honour it, but several
deliberately diverge:

| File | Convention | Notes |
|---|---|---|
| `biotsavart.py` | `[p, j, l]` (correct) | `biot_savart_dB_by_dX`, swap-axes at line 477 |
| `wireframe.py` | `[p, k, m]` (component-first) | documented at module head (line 27-40); claims compatibility with C++ kernel storage and notes the abstract convention is the same array with swapped labels |
| `dipole_field.py` | `[p, j, k] = d B_j / d x_k` (component-first) | line 246-247 docstring |
| `circular_coil.py` | `[p, j, l]` (correct, line 551-555) | OK |
| `analytic_pure_fields.py::toroidal_dB` / `toroidal_dA` / `poloidal_dB` | `[p, l, j]` (component-first) | line 331-340 docstring explicitly says it's "intentional" to match the CPU oracle storage |
| `analytic_pure_fields.py::mirror_dB` | `[p, j, l]` (correct, line 644-645) | OK |
| `analytic_fields.py::dommaschk_dB` | `[k, p, i, j]` per docstring at line 663 | uses `i` = deriv axis, `j` = component — same as biotsavart |
| `analytic_fields.py::reiman_dB` | `[p, i, j]` per docstring at line 884-887 | matches biotsavart |
| `magnetic_axis_helpers.py` | consumes `[j, l]` at line 227 | OK; documents the convention explicitly |

The split is documented and load-bearing for parity with the C++ oracle
(see CLAUDE.md "Code Review History → Confirmed NOT bugs"). For an
auditor reading the codebase top-down, the **best fix is to either (a)
rename the storage-divergent functions, e.g.
`toroidal_dB_cpu_oracle_order`, or (b) add a single-line in each
function docstring stating its axis convention without ambiguity**.
Several already do this; the dipole/wireframe pair is the noisiest.

### §A-05. `__init__.py` lazy-load safety — confirmed pass

`jax_core/__init__.py` is a pure import chain (no top-level
`import jax`, no work). Re-exports are kept synchronised between
`from … import …` and `__all__`. The expected import-time side
effects are limited to triggering the registered
`jax.tree_util.register_dataclass` calls in `specs.py`,
`analytic_pure_fields.py`, `boozer_analytic.py`, `tracing.py`,
`pm_optimization.py`, `interpolated_field.py`, `boozer_fixed_state.py`,
`dipole_field.py`, `circular_coil.py`. These are idempotent.

### §A-06. Public-symbol naming — minor inconsistency

- `circular_coil.py:CircularCoilSpec` and `analytic_pure_fields.py`
  Specs include the `Spec` suffix.
- `boozer_analytic.py::BoozerAnalyticFrozenState` and
  `boozer_fixed_state.py::BoozerRadialFixedState` use `FrozenState`
  instead.
- `interpolated_boozer_field.py::InterpolatedBoozerFieldFrozenState`
  uses `FrozenState`.
- `interpolated_field.py::InterpolatedFieldSpec` uses `Spec`.

Reduces grep-ability when looking for "is X a frozen pytree?".
Suggested: pick one suffix (`Spec` works) and stick with it.

### §A-07. Docstring source references — generally strong

`magnetic_axis_helpers.py` (Greene 1979 eq. 13), `_elliptic.py`
(Numerical Recipes 6.11), `tracing.py` (Hairer DOPRI5),
`pm_optimization.py` (Bouchala et al. 2014), `framedcurve.py` (Singh
et al. 2020), `analytic_fields.py` (Dommaschk 1986, Reiman 1986),
`finitebuild.py` (Singh et al. 2020) all cite the underlying physics
papers. Boozer kernels (`boozer_analytic.py`,
`boozer_radial_interp.py`) cite the upstream CPU oracle line numbers.

---

## B. JAX best-practice compliance

### §B-01. `jax.lax.while_loop` is not reverse-mode differentiable

**Severity:** HIGH
**Category:** jax-best-practice
**Files / lines:**

- `magnetic_axis_helpers.py:515` (`_integrate_tangent_map` driver)
- `tracing.py:1234` (`trace_fieldline`)
- `tracing.py:1778` (`trace_guiding_center`)
- `tracing.py:1973`/`2799` (`trace_guiding_center_boozer` and full-orbit)
- `tracing.py:3342` (`trace_fullorbit`)
- `tracing.py:858` (`bracket_root_jax`)

**What's wrong:** all of these wrap an unbounded-iteration adaptive
integrator (or Illinois root finder) in `jax.lax.while_loop`. JAX's
`while_loop` HLO lowering does not support reverse-mode AD — calling
`jax.grad` / `jax.vjp` on any of these functions raises
`TypeError: Reverse-mode differentiation does not work for lax.while_loop.`

The docstring on
`magnetic_axis_helpers.on_axis_iota_rk` (line 16) advertises:

> "the entire iota computation is JIT-able and differentiable through
> field DOFs as long as the user supplies a JAX-traceable
> field-evaluation callback"

This is **not** correct for reverse-mode AD. It is true only for
forward-mode (`jax.jvp` / `jax.jacfwd`), which JAX does support
through `while_loop`.

**Why it matters:** any downstream code that builds an objective on
top of these integrators and reaches for `jax.grad` (the dominant API
for outer optimization) will fail at trace time, not at runtime. The
JAX-port plan's whole purpose is reverse-mode AD up to the outer
optimizer, so this is squarely on the critical path.

**Suggested fix:**

1. Documentation: amend `on_axis_iota_rk` and the `trace_*` docstrings
   to state explicitly that they are forward-mode differentiable only.
   Pin a doc note in `docs/source/jax_acceptance.rst` so the contract
   is observable.

2. Code: if reverse-mode is in scope, replace `while_loop` with a
   bounded `lax.scan` of length `max_steps` and mask inactive
   iterations after termination. This is exactly the same trick the
   trace drivers already use for the trajectory buffer (cf.
   `_finalize_trajectory_rows`). The pattern composes cleanly with
   reverse-mode at the cost of always doing `max_steps` worth of work
   (and `O(max_steps)` extra memory for the saved residuals); for the
   axis-iota integration `max_steps = 10000` would only cost ~5 MB of
   residual tape per call.

3. Alternative: wrap the forward solve in a `jax.custom_vjp` with a
   manual adjoint, following the IFT / continuous-adjoint approach
   already established in `geo/boozersurface_jax.py` for the Boozer
   adjoint. This is the production-grade path.

### §B-02. Split axis convention between `analytic_pure_fields` and the rest

**Severity:** HIGH
**Category:** simsopt-convention + jax-best-practice
**File:** `src/simsopt/jax_core/analytic_pure_fields.py`

**What's wrong:** Within a single module, `toroidal_dB`, `toroidal_dA`,
and `poloidal_dB` document and return arrays in `[p, l, j]` order (axis
1 = component, axis 2 = derivative), whereas `mirror_dB` in the same
module returns `[p, j, l]` (axis 1 = derivative, axis 2 = component).
The module-level docstring at line 16-23 acknowledges this split:

> "First-derivative layouts mirror the upstream CPU classes **as they
> actually store the array**, not the abstract simsopt convention
> documented in `CLAUDE.md`."

The relevant per-point assemblers are at:

- `_toroidal_dB_pointwise` line 175-195 ends with `.T`
- `_toroidal_dA_pointwise` line 266-286 ends with `.T`
- `_poloidal_dB_pointwise` line 394-473 ends with `.T`
- `_mirror_dB_pointwise` line 557-609 does **not** transpose.

**Why it matters:** there is no public-facing flag that tells the user
which layout a given function uses. Two adjacent functions in the same
file disagree, and the reasoning ("CPU oracle storage layout") is an
implementation detail. The CLAUDE.md tensor convention is by reference
the contract; this module fights the contract.

The deviation is justified by the parity-test contract
(`direct_kernel` lane). Once a non-trivial pipeline composes the JAX
toroidal output into a downstream module that assumes the abstract
convention, the result is silently wrong.

**Suggested fix:** Two options.

1. Rename the divergent helpers to make the storage explicit:
   `toroidal_dB_cpu_oracle_layout`, with `toroidal_dB`
   transposing back to `[p, j, l]` before returning. The parity test
   then asserts equality after the **caller's** transpose, not before.
2. Leave the storage as is but raise a `DeprecationWarning` if the
   caller doesn't pass an explicit `axis_convention=` kwarg. This is
   less invasive but more surface area.

Either way, every public function in `analytic_pure_fields.py` should
print the layout in its **one-line summary**, not just the longer
docstring.

### §B-03. Mutable `dict` inside a `frozen=True` dataclass

**Severity:** HIGH
**Category:** jax-best-practice + immutability
**File:** `src/simsopt/jax_core/interpolated_boozer_field.py:164-235`

**What's wrong:**

```python
@dataclass(frozen=True)
class InterpolatedBoozerFieldFrozenState:
    specs: dict          # ← mutable, mutated in place
    nfp: int
    stellsym: bool
    extrapolate: bool
    period: float
    s_range: tuple
    theta_range: tuple
    zeta_range: tuple
    degree: int
```

The docstring at lines 178-188 explicitly describes the mutation
contract:

> "**Mutability contract**: the dataclass attributes are frozen — they
> cannot be reassigned after construction. However, the underlying
> `specs` dict object is **mutated in place** by
> `InterpolatedBoozerFieldJAX._ensure_spec` when a scalar is lazy-built
> after construction."

The class is intentionally **not** registered as a JAX pytree (line
204-207), so the dict mutation does not corrupt a pytree leaf
listing. But:

- `evaluate_scalar` (line 657) reads `state.specs[name]` inside what
  looks like (and is documented to be) a JIT-friendly evaluation path.
  Concretely, the result of `state.get(scalar_name)` is a
  `RegularGridInterpolant3DSpec` whose **fields are NumPy arrays**;
  these are converted to device arrays inside `evaluate_batch`. If
  another thread (or the wrapper's lazy-build path) replaces the dict
  entry while a JIT trace is in flight, the trace captures a stale
  reference.

- The CLAUDE.md guardrails explicitly require IMMUTABLE state.

**Why it matters:** the mutation is not protected by a lock or
re-traced explicitly. The wrapper class in
`field/boozermagneticfield_jax.py` (out of scope here) owns the
discipline. If a future contributor parallelises construction of the
specs (e.g. one thread per scalar), the dict races.

**Suggested fix:**

- Pre-build every requested scalar at `freeze_interpolated_boozer_field_state`
  (the API already accepts an explicit `scalars=` tuple). The function
  body at line 540-550 builds them upfront. The lazy-build is in the
  wrapper class, not here, so the kernel-layer dataclass should be
  effectively read-only.
- Rename the field to `_specs_mutable` or wrap it in `MappingProxyType`
  so the in-place mutation is at least surface-visible.
- Document the safe-write barrier in the wrapper.

### §B-04. `jnp.linalg.eig` is a host-callback on jaxlib 0.10.0

**Severity:** HIGH
**Category:** jax-best-practice
**File:** `src/simsopt/jax_core/magnetic_axis_helpers.py:598`

```python
M = y_final.reshape((2, 2))
evals, _ = jnp.linalg.eig(M)
```

**What's wrong:** in jaxlib 0.10.0, `jnp.linalg.eig` is implemented
via LAPACK's `geev` through `jax.pure_callback` on every non-Hermitian
input. For a 2x2 matrix the callback overhead dominates the actual
work, and (critically) the callback materialises the matrix on host.
Under `transfer_guard("disallow")` (which `_math_utils.py` line 11
goes out of its way to support via `maybe_initialize_distributed_jax`)
the callback will succeed silently because the transfer guard does not
intercept pure callbacks, but on a GPU device the round-trip cost is
~200 µs of dead time per call.

**Why it matters:** every call to `on_axis_iota_rk` (item 14
follow-up) ends in this LAPACK call. The whole purpose of the JAX
port is to keep the optimization hot path device-resident.

**Suggested fix:** the matrix is 2x2; the eigenvalues have a closed
form. Replace lines 597-602 with:

```python
M = y_final.reshape((2, 2))
trace = M[0, 0] + M[1, 1]
det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
# Eigenvalues of a 2x2: (trace ± sqrt(trace²-4 det)) / 2
disc = trace * trace - 4.0 * det
# Complex sqrt — disc may be negative for the conjugate-pair case
# that produces a physical iota.
eig_real = 0.5 * trace
eig_imag = 0.5 * jnp.sqrt(jnp.maximum(-disc, 0.0))
iota = jnp.arctan2(eig_imag, eig_real) * nfp_arr / two_pi
```

(The same trick is referenced as "we replicate the same logic" in the
docstring at line 56-58, so the closed form is already canonical for
this matrix family.) This drops the host callback entirely and makes
the whole `on_axis_iota_rk` device-resident.

### §B-05. `_make_kernel` LRU cache keyed on platform — wasted compilations

**Severity:** HIGH (perf only)
**Category:** jax-best-practice
**File:** `src/simsopt/jax_core/biotsavart.py:427-516`

```python
@lru_cache(maxsize=32)
def _make_kernel(
    integrand_key, diff_mode, coil_cs, quad_bs, point_cs,
    point_vma_axis_name, platform,
):
    del platform
    ...
```

**What's wrong:** the cache key includes `platform` (line 529 reads
`jax.default_backend()` at every call). The factory body explicitly
discards it (line 449 `del platform`). The intent — per the docstring
at lines 444-448 — is for the cache to "stay aligned with the device
the next call will use". That's correct as a *cache invalidation*
strategy, but the cache entries are identical Python closures: XLA
will still compile them once per platform.

**Why it matters:** an interactive session that does
```python
B_cpu = biot_savart_B(points, gammas, gammadashs, currents)
# … switch backend …
B_gpu = biot_savart_B(points, gammas, gammadashs, currents)
```
will compile two distinct kernel closures and burn two slots out of
the 32-cap LRU. With six diff modes and several typical tunings, the
cap of 32 is reached easily, and the LRU evicts older kernels
including the original CPU kernel. The fix is to compose the
platform into the JIT key separately:

```python
@lru_cache(maxsize=32)
def _make_kernel_inner(...):   # no platform
    ...

def _get_kernel(integrand_key, diff_mode, *, point_vma_axis_name=None):
    coil_cs, quad_bs, point_cs = _read_tuning_config()
    kernel = _make_kernel_inner(integrand_key, diff_mode, coil_cs, quad_bs, point_cs, point_vma_axis_name)
    # JAX will pick the right compiled-executable for the active device at call time.
    return kernel
```

The XLA compilation cache (keyed on the active device internally) does
the right thing. The Python-level LRU only needs to dedupe identical
factory bodies.

### §B-06. `_safe_radius_squared` silently floors `r²` at 1e-60

**Severity:** MEDIUM
**Category:** jax-best-practice + simsopt-convention
**File:** `src/simsopt/jax_core/biotsavart.py:119-131`

The helper documents the divergence from C++ behavior, but the
implication is hidden from the public functions:

```python
def _safe_radius_squared(diff):
    r2 = jnp.sum(diff * diff, axis=-1)
    return jnp.maximum(r2, _float64_scalar(r2, 1e-60))
```

The choice of `1e-60` is interesting: it places the floor below the
`1e-50` smallest-resolvable-distance regime but above
`np.finfo(np.float64).tiny ≈ 5e-324`. At `r² = 1e-60`, `rinv³ =
1/r³ = 1e+90`, which is comfortably inside float64 range.

The risk surfaces in three places:

1. `biot_savart_B`, `_B_vjp` callers cannot distinguish a coil-on-point
   evaluation from a genuinely small `r²`. The C++ oracle would
   propagate NaN/Inf for the former.
2. Gradients through this clamp are zero when the floor is active.
   That means `dB/dgamma` near a point-on-coil location reports
   spuriously-bounded sensitivities, with the discontinuity at exactly
   `r² = 1e-60`.
3. The clamp is **not** documented in the `biot_savart_B` /
   `biot_savart_B_vjp` public docstrings; only the private helper has
   the explanation.

**Suggested fix:** lift the documentation into the public docstrings
and the `__init__.py` re-export comment. Consider exposing a
`r2_floor: float = 1e-60` argument so consumers in the test layer can
opt back into the unsafeguarded form to match the C++ oracle when they
want.

### §B-07. `meta_fields` make spec parameters static (not autodiff-able)

**Severity:** MEDIUM
**Category:** jax-best-practice
**Files / lines:**

- `specs.py:107` — `CurveHelicalSpec.meta_fields=["order", "m", "ell", "R0", "r"]`
- `circular_coil.py:108` — `CircularCoilSpec.meta_fields=["r0", "center", "Inorm", "normal", "normal_kind"]`
- `analytic_pure_fields.py:91-95` — `ToroidalFieldSpec` puts `R0`, `B0`
  in `meta_fields` with no `data_fields`
- similar pattern in `PoloidalFieldSpec`, `MirrorModelSpec`,
  `BoozerAnalyticFrozenState` (data_fields), `DommaschkSpec`,
  `ReimanSpec`

**What's wrong:** when a parameter is declared as a `meta_field`,
`jax.tree_util.register_dataclass` treats it as **static** — any change
forces a re-trace. A consumer doing
```python
jax.grad(lambda r0: circular_coil_B(CircularCoilSpec(r0=r0, ...)))(jnp.asarray(1.0))
```
will fail because `r0` is treated as part of the JIT cache key, not
data. The current path of routing through `_scalars` (line 515-525)
materialises everything as `jax.Array` *before* hitting the JIT
boundary, so within a single `circular_coil_B` call the scalars are
data — but the spec itself is not differentiable.

`CircularCoilSpec` particularly is unusual because `r0`/`Inorm` are
canonical optimization knobs in stellarator design. The current
factoring works for forward evaluation but blocks reverse-mode AD
through the spec.

**Suggested fix:** prefer `data_fields` for any quantity that might
be optimized; reserve `meta_fields` for static integer shape
parameters (`order`, `mpol`, `ntor`, etc.). For specs that need both,
split the dataclass into a `…StaticSpec` and a `…RuntimeSpec` pair
(some specs in `specs.py` already do this, e.g.
`CurveXYZFourierSpec` puts `dofs` / `quadpoints` in data and only
`order` in meta — that is the recommended pattern).

For `CircularCoilSpec` specifically: move `r0`, `Inorm`, `center` to
`data_fields`. Keep `normal_kind` (string) in `meta_fields`.

### §B-08. `scalar_potential_rz.py` "tiny ghost term" pattern

**Severity:** MEDIUM
**Category:** jax-best-practice
**File:** `src/simsopt/jax_core/scalar_potential_rz.py:18, 35-62`

```python
_TINY_SYMPY_TERM = sp.Float("1e-30") * _PHI_SYMBOL * _R_SYMBOL * _Z_SYMBOL
...
b_eval = lower_sympy_expressions(
    (
        phi_R + _TINY_SYMPY_TERM,
        phi_Phi_over_R + _TINY_SYMPY_TERM,
        phi_Z + _TINY_SYMPY_TERM,
    )
)
```

**What's wrong:** the comment in `_sympy_to_jax.py` (line 138-140)
explains:

```python
def _constant_like(value: sp.Expr, reference: jax.Array) -> jax.Array:
    return jnp.asarray(float(value), dtype=reference.dtype) + jnp.zeros_like(reference)
```

When SymPy folds an expression to a numeric constant (e.g.
`phi_R.diff(_R_SYMBOL)` for a constant potential), the SSOT lowering
would lose the array shape. The `_TINY_SYMPY_TERM` defeats the
constant-fold by injecting `phi * R * Z` dependency on every term.

**Why it matters:**

1. A systematic, **deterministic** ~`1e-30 * R * Z * phi` is added to
   every B and dB output. On a stellarator of radius ~1 m at maximum
   excursion 0.5 m, `|R * Z * phi| ~ π/2`, and the ghost is
   ~`5e-30 T`. That's below the float64 noise floor at the typical
   `|B| ~ 1` T scale, but is a real bias that will show up in
   regression tests with tolerances at `1e-20` or tighter.
2. The hack is implicit in the source; the SSOT layer
   (`_sympy_to_jax.py`) could absorb it (always broadcast scalar
   expressions to the input shape) without the caller knowing.

**Suggested fix:** in `_sympy_to_jax.py::_constant_like`, the
existing `+ jnp.zeros_like(reference)` already broadcasts to the
input shape. There is no need for the ghost term in
`scalar_potential_rz.py`. Verify by removing
`_TINY_SYMPY_TERM` and re-running the test suite — if the test fails
because of constant-fold loss, the fix belongs inside `_eval_expr` or
`_constant_like`, not at the call site.

### §B-09. `jnp.atan2` vs `jnp.arctan2` naming inconsistency

**Severity:** LOW (nit)
**Category:** jax-best-practice
**Files:** `analytic_pure_fields.py:380, 381, 402, 403, 545, 565`;
`dipole_field.py:322, 323`. All other files use `jnp.arctan2`.

`jnp.atan2` is the SciPy-style name and is supported in JAX 0.10.0
as an alias. `jnp.arctan2` is the NumPy-compatible name. The
inconsistency is purely cosmetic. Recommend converging on
`jnp.arctan2` (NumPy parity is the prevailing style in this
codebase).

### §B-10. `dipole_field.py` `jnp.where` ordering at the origin

**Severity:** LOW
**Category:** jax-best-practice
**File:** `src/simsopt/jax_core/dipole_field.py:322`

```python
phi = jnp.where(radius == 0.0, 0.0, jnp.atan2(y, x))
```

This is the wrong order for the "double-where" NaN-safe pattern. The
`jnp.atan2(0.0, 0.0)` is evaluated regardless of the `where` branch.
JAX evaluates it to `0.0` per IEEE-754 specifications, so the primal
is harmless. But the **gradient** of `atan2` at `(0, 0)` is undefined
(`d atan2 / dy = x/(x²+y²)` blows up). Reverse-mode AD through this
will produce NaN gradients at the origin even though the `where`
selected the safe branch.

**Why it matters:** the C++ oracle does not have an origin guard here
either — coils sitting on the origin are unphysical. But the JAX
gradient story is worth documenting.

**Suggested fix:** use the canonical double-where pattern:

```python
radius_safe = jnp.where(radius == 0.0, 1.0, radius)  # any nonzero placeholder
# proceed with radius_safe; atan2 still operates on x, y which we leave alone
phi = jnp.where(radius == 0.0, 0.0, jnp.arctan2(y, x))
```

Alternatively, document that dipoles must not sit at the cylindrical
axis. This was never going to be a sharp issue in practice.

### §B-11. `compensated_sum_flat` is not reverse-mode differentiable

**Severity:** LOW
**Category:** jax-best-practice
**File:** `src/simsopt/jax_core/reductions.py:75-96`

The Kahan compensated summation uses `lax.fori_loop` (line 94), which
is not reverse-mode differentiable. Most reductions go through
`pairwise_sum_flat` / `pairwise_sum_axis` (which use `jnp.sum`
internally and are differentiable). `compensated_sum_flat` is gated
behind `reduction_mode="strict_oracle"` per
`scalar_square_sum` (lines 116-117), so it is only used in parity-lane
diagnostics. Worth a one-line warning in the docstring.

### §B-12. Python `for` loops inside `while_loop` bodies are unrolled

**Severity:** LOW (intentional but worth flagging)
**Category:** jax-best-practice
**Files:** `tracing.py:1059-1112, 1135-1166`; analogous loops in the
other trace drivers.

```python
def scan_phis(args):
    ...
    for i in range(num_phis):
        ...
```

`num_phis` is bound to `int(phis_arr.shape[0])` (line 948) — static
at trace time. Each Python iteration adds its own subgraph to the
`scan_phis` body. A user passing `phis = jnp.linspace(0, 2*pi, 100)`
makes the body grow ~100x without warning. The same holds for the
`stopping_criteria` enumerate-loop (line 1135).

**Why it matters:** trace time grows linearly in `num_phis` and
`len(stopping_criteria)`; compile time grows worse than linearly for
the dependent operations. Realistic phi counts are ≤16, so this is
fine in practice. Worth a doc note that the buffer is unrolled
statically.

### §B-13. `boozer_radial_interp.py` polymorphic `inverse_fourier_transform_*` dispatch

**Severity:** LOW (style)
**Category:** jax-best-practice
**File:** `src/simsopt/jax_core/boozer_radial_interp.py:472-491,
529-546`

The two `inverse_fourier_transform_{odd,even}` wrappers do
`if kmns.ndim == 1: ... if kmns.ndim == 2: ...`. The dispatch happens
at the Python layer and is fine because `kmns.ndim` is static. Both
underlying kernels (`_1d` / `_2d`) are independently JIT'd. The Python
dispatcher won't trace through; if a user `jit`s the dispatcher
directly (`jax.jit(inverse_fourier_transform_odd)`) the `if .ndim`
check is **static** under JIT (shape is known) so this works.

A cleaner pattern is to make the dispatcher non-jittable on purpose
and add a docstring note: "If you need a `jit` boundary at this
level, jit the `_1d` or `_2d` variant directly." Otherwise OK.

### §B-14. PRNG discipline — `sampling.py` is good

**File:** `src/simsopt/jax_core/sampling.py`. Excellent: every public
function takes `key: jax.Array` as the first required argument.
`jax.random.uniform(key, ...)` is the only RNG used.
`np.random.default_rng(seed)` is used **only** in
`regular_grid_interp.estimate_error` (line 728), which is a host-side
diagnostic helper. Good separation.

### §B-15. `donate_argnums` is not used anywhere

**Severity:** LOW
**Category:** jax-best-practice

Grep `donate_argnums` and `donate_argnames` across the kernel layer:
zero hits. The `lax.scan` loops in `pm_optimization.py` (e.g. line
2519) carry large state buffers (`(N, 3)`) for `n_steps` iterations.
Donating the carry would save the per-iter copy. Likely not the
bottleneck, but worth profiling for the production permanent-magnet
optimizer.

---

## C. Cross-package dependency / layering findings

### §C-01. `jax_core/` transitively imports `Optimizable` and the simsoptpp shim

**Severity:** BLOCKER (CLAUDE.md SSOT contract)
**Category:** simsopt-convention
**Files / lines:**

| jax_core file | imports | leaks |
|---|---|---|
| `curve_geometry.py:12-21` | `..geo.curve.gamma_curve_on_surface`, `..geo.curvehelical.curve_helical_pure`, `..geo.curveplanarfourier.curveplanarfourier_pure`, `..geo.curverzfourier.curverzfourier_pure`, `..geo.curvexyzfourier.{jaxfouriercurve_pure, jaxfouriercurve_geometry_pure}` | `simsopt.geo.curve` imports `_simsoptpp.has_simsoptpp_symbol`, `_simsoptpp.sopp_namespace`, `simsopt._core.optimizable.Optimizable`, `simsopt._core.derivative.Derivative` |
| `curve_geometry.py:152` (lazy) | `simsopt.geo.orientedcurve.centercurve_pure` | same chain |
| `curve_geometry.py:188` (lazy) | `simsopt.geo.curvexyzfouriersymmetries.jaxXYZFourierSymmetriescurve_pure` | same |
| `magnetic_axis_helpers.py:69` | `..geo.curverzfourier.curverzfourier_pure` | `Optimizable` chain |
| `specs.py:1572` (lazy) | `..geo.surface_fourier_jax.stellsym_scatter_indices` | same |
| `surface_fourier.py:7-25` | `..geo.surface_fourier_jax.{surface_area, surface_*_from_dofs}` | same |
| `surface_henneberg.py:36` | `..geo.surface_fourier_jax.{surface_area, surface_volume}` | same |
| `objectives_flux.py:32-35` | `..objectives.integral_bdotn_jax.{integral_BdotN, residual_BdotN}` | OK (this is JAX-only, no Optimizable), but inverts the layering arrow |
| `tracing.py:75-91` | `..field.boozermagneticfield_jax.{BoozerRadialInterpolantFrozenState, _eval_*}` | the boozer adapter is in `field/`, which imports from `Optimizable` |

**What's wrong:** the CLAUDE.md jax-core layout section claims:

> "JAX modules live alongside C++ counterparts. They do NOT import
> simsoptpp."

That is technically true at the leaf level — none of these modules
`import simsoptpp` directly. **But** they import from `simsopt.geo` /
`simsopt.field`, which import `Optimizable` from `simsopt._core`,
which is the layer that ultimately calls into `_simsoptpp`. The
"no simsoptpp dependency" contract is only honoured because of the
runtime guards in `simsopt/geo/_simsoptpp.py` (the placeholder
type factory), not because the JAX core is genuinely decoupled.

**Why it matters:**

1. Packaging: shipping `simsopt.jax_core` as a standalone JAX library
   is impossible without also shipping `simsopt.geo`, `simsopt._core`,
   and the simsoptpp shim. The CLAUDE.md story implies the JAX core
   could be lifted out cleanly. It cannot.
2. Import cost: a `import simsopt.jax_core.biotsavart` pulls every
   `Optimizable` class through transitively. Cold-start latency on
   GPU notebooks is non-trivial.
3. Cycle risk: the memory note `project_curve_jax_core_import_cycle`
   already records that `simsopt.geo.curve ↔ simsopt.jax_core` forces
   lazy imports inside helpers. Each new helper that needs a "pure"
   kernel from `geo/` adds another lazy import.

**Suggested fix:** move the pure helpers downstream of `jax_core/`:

- Relocate `gamma_curve_on_surface`, `*_pure` curve kernels,
  `surface_fourier_jax.*` SSOT functions, and the
  `integral_BdotN` / `residual_BdotN` reducers **into `jax_core/`**.
  Have the `Optimizable` adapter modules in `geo/` and `field/`
  import **down** from `jax_core/`, not up.
- This is consistent with the existing pattern in
  `jax_core/finitebuild.py` (uses `..specs`), `jax_core/biotsavart.py`
  (uses `..backend`), `jax_core/regular_grid_interp.py` (no
  cross-package imports), and the bulk of `jax_core/`.
- The migration is mechanical: `git mv simsopt/geo/curvexyzfourier.py
  simsopt/jax_core/curvexyzfourier_pure.py` (keeping only the
  `*_pure` functions), then update import sites. About 9 files affected.

This is the single largest-impact change in this audit; it deserves a
plan + PR rather than a one-line fix.

### §C-02. `simsopt.backend` is the only "up" import that is correct

**File:** `src/simsopt/jax_core/biotsavart.py:21-24`,
`src/simsopt/jax_core/_math_utils.py:11-13`,
`src/simsopt/jax_core/sharding.py:14-15`. These three import
`simsopt.backend` and `simsopt.backend.runtime`.

`simsopt.backend` is a thin configuration / RNG-key / device-init
module that does not pull `simsoptpp`. This is the canonical
"infrastructure" arrow and is fine. Worth documenting in CLAUDE.md
that `simsopt.backend` is the explicit upstream allowed import.

---

## D. Smaller observations (low severity)

### §D-01. `_device_scalars.py::device_one` is a clever but obscure pattern

**File:** `src/simsopt/jax_core/_device_scalars.py:9-19`

```python
def device_one(reference: jax.Array) -> jax.Array:
    return jnp.exp(jnp.sum(reference - reference))
```

This produces a device-resident `1.0` of the same dtype as `reference`
without any host-to-device transfer. The intent is sound (everything
that could trip the transfer guard is `reference - reference`). But:

1. `exp(sum(0))` is a numerical operation that depends on the
   compiler folding it to `1.0`. On most backends XLA folds the
   constant, but it's not guaranteed.
2. `two_pi(reference)` follows up with `pi = arccos(-device_one(reference))`,
   so the value of π is computed from a chained
   `exp(sum(diff))` plus `arccos`. This is intentional (avoid Python
   literal `2 * np.pi` crossing the boundary). It's a recurring
   pattern in `boozer_analytic.py`, `surface_rzfourier.py`,
   `objectives_flux.py`, `interpolated_boozer_field.py`, etc.

**Suggested fix:** none, but document at the module head why
`device_one` exists and what would break without it (the answer is:
`jax.transfer_guard("disallow")`).

### §D-02. `analytic_fields.py` `lru_cache(maxsize=None)`

**File:** `src/simsopt/jax_core/analytic_fields.py:251, 528, 559, 840, 852`

Five `@lru_cache(maxsize=None)`-decorated factories with `(m, n)` /
`(k_theta, m0_symmetry)` static arguments. Unbounded cache. For a
typical Dommaschk simulation with ≤30 modes this is fine, but worth
noting that an interactive sweep that varies mode indices many times
can leak memory. Better to bound at `maxsize=128` or similar.

### §D-03. `_TINY_SYMPY_TERM` constant in `scalar_potential_rz.py`

Already covered in §B-08.

### §D-04. `_basis_values` in `regular_grid_interp.py`

**File:** `src/simsopt/jax_core/regular_grid_interp.py:494-516`

The Lagrange basis evaluation uses an `eye + (1 - eye)` trick to
turn off the diagonal in the product. Clever and correct, but
quadratic in `degree`. For the typical `degree ∈ {2, 3, 4}` it's
fine; if degree grows, a recursive `lax.scan` formulation would be
better.

### §D-05. `surface_classifier.py:78` raises inside a JAX-traceable closure

```python
def classify(xyz: jax.Array) -> jax.Array:
    xyz_arr = jnp.asarray(xyz, dtype=jnp.float64)
    was_single = xyz_arr.ndim == 1
    ...
    if xyz_arr.ndim != 2 or xyz_arr.shape[-1] != 3:
        raise ValueError(...)
```

The `if` is a static check (`.ndim` and `.shape[-1]` are known at
trace time), so this is correct under `jit`. It would surface a
`ConcretizationTypeError` if a tracer with unknown shape ever reached
it, but JAX never produces such tracers for this entry path.

### §D-06. `tracing.py:_finalize_trajectory_rows` uses `fori_loop` to backfill

**File:** `src/simsopt/jax_core/tracing.py:620`

```python
traj_padded = jax.lax.fori_loop(0, max_steps + 1, fill_padding, traj_final)
```

`fori_loop` is not reverse-mode differentiable (same as `while_loop`
internally), but the operation here is purely the final-state
backfill of the trajectory tape. Differentiation through it is
unlikely to be needed because the trajectory ends with a discrete
"stop" event.

Worth a one-line note: "this final-backfill loop is intentionally
last-write-wins and not differentiable; gradients should be taken
through the live prefix only."

### §D-07. `boozer_fixed_state.py:204` Python `for` loop

**File:** `src/simsopt/jax_core/boozer_fixed_state.py:198-206`

```python
acc = gathered[..., 0, :]
for icoeff in range(1, gathered.shape[-2]):
    acc = acc * local + gathered[..., icoeff, :]
return acc
```

Horner's rule, statically unrolled (gathered.shape[-2] is the static
poly degree). Correct and fast.

### §D-08. `pm_optimization.py:1855` `lax.cond` inside `lax.scan`

A few of the GPMO solvers nest `lax.cond` inside `lax.scan` for the
backtracking removal step. This is standard JAX; `cond` is
reverse-mode safe.

### §D-09. `wireframe.py` documents axis layout consistent with the C++ oracle

The wireframe module is one of the cleanest in the package: every
function has a tight, parity-anchored docstring; the closed-form
Jacobian is transcribed from the C++ rather than derived through
autodiff to guarantee bit-identical parity. No issues.

### §D-10. `sharding.py` uses modern `NamedSharding` / `Mesh` API

No `pmap` anywhere in the kernel layer. The pattern follows the JAX
shard-map example from the 0.4+ release notes. The `_clear_sharding_caches`
hook registered with `register_backend_cache_clear` (line 427) is the
right cleanup pattern.

### §D-11. `magneticfield_composition.py` is a tight pass-through layer

Twelve thin one-liners that sum or scale children. Pure, JIT-safe,
no concerns. The one observation is that all primitives accept a
generic `KernelCallable` (Python callable closing over its own
frozen spec); this is consistent with the JAX-friendly composition
style.

### §D-12. `mhd_reductions.py:28-40` mixes NumPy and JAX

```python
xm = np.asarray(xm_b)
xn = np.asarray(xn_b) / nfp
...
symmetric_indices = np.nonzero(symmetric)[0]
nonsymmetric_indices = np.nonzero(np.logical_not(symmetric))[0]
```

The mode-index helper is a **host-side** function (returns NumPy
arrays). The docstring at line 56-58 correctly says:

> "Compute it outside JAX transforms with
> `boozer_quasisymmetry_mode_indices` and pass the resulting
> fixed-shape index arrays here."

This is the right partition. The NumPy use is justified.

### §D-13. `tracing.py` uses `jnp.linalg.solve` nowhere

I checked: only `eig` in `magnetic_axis_helpers.py`. No
`jnp.linalg.solve` / `inv` calls anywhere in the kernel layer.

### §D-14. `pm_optimization.py` long file but consistent

2522 LOC. The pattern is identical across the four GPMO variants:
spec dataclass + result dataclass + `*_candidate_costs` (a single
reduction over a fixed-shape array) + `*_step` (a `lax.scan` body) +
`*_solve` (the outer `lax.scan` of length `K`). Each variant tracks
the same set of trace columns. The duplication is **intentional** —
each variant has subtly different state — and the code is readable.

---

## Positive notes

- **PRNG hygiene** is excellent. `sampling.py` takes explicit
  `jax.random.PRNGKey`. No global RNG anywhere in the kernel layer.
- **Numerical safety** is generally well-thought-out. `_math_utils.py`
  provides `unit_vector_axis_last` with a custom JVP that handles
  zero-length vectors correctly. `_elliptic.py` provides a
  fully traceable Carlson-form elliptic integral, replacing the
  missing `jax.scipy.special.ellipk` in jaxlib 0.10.0.
- **`custom_jvp` / `custom_vjp` discipline** is sound where used:
  `_math_utils.unit_vector_axis_last` has a proper projected-tangent
  JVP; `_math_utils.inf_with_nan_jvp` defines a controlled gradient
  for the failure marker; `_math_utils.explicit_inv` and
  `explicit_rsqrt` define safe inverses. All four are used downstream
  in `biotsavart.py`, `surface_rzfourier.py`, etc.
- **Sharding** uses the modern `NamedSharding(Mesh, P(...))` API
  (no `pmap`).
- **C++ parity docstrings** are consistent. Most kernels list the
  upstream C++ file and line range they mirror, e.g.
  `boozer_radial_interp.py:236-241`,
  `pm_optimization.py:106-115`,
  `regular_grid_interp.py:1-26`,
  `interpolated_boozer_field.py:91-150`,
  `tracing.py:139-204`.
- **Closure capture in jit factories** is well-handled: every
  factory that builds a `@jax.jit` closure does so inside an
  `lru_cache`-keyed function so that closure identity is stable for
  the JAX trace cache. Examples:
  `biotsavart.py::_make_kernel`,
  `analytic_fields.py::_dommaschk_B_multimode_kernel`,
  `analytic_fields.py::_dommaschk_dB_multimode_kernel`,
  `analytic_fields.py::_reiman_B_kernel`,
  `analytic_fields.py::_reiman_dB_kernel`.
- **`scan_phis` / `apply_criteria` body-unrolling** in
  `tracing.py:1059, 1135` is the correct JAX pattern for
  trace-time-static control flow inside a `while_loop` body.

---

## Final verdict per file group

| Group | Files | Verdict |
|---|---|---|
| Math / device helpers | `_device_scalars.py`, `_math_utils.py`, `_elliptic.py`, `_sympy_to_jax.py` | **READY** — clean, well-documented, no issues |
| Pure analytic fields | `analytic_fields.py` (Dommaschk + Reiman) | **READY** with caveat — module-level docstring should call out that the JAX path differs in the dB axis convention vs the broader simsopt-jax convention; otherwise solid |
| Closed-form fields | `analytic_pure_fields.py` (Toroidal/Poloidal/Mirror) | **NOT READY** — fix §B-02 (split axis convention) before declaring "convention complete" |
| Biot-Savart | `biotsavart.py`, `biotsavart_cpu_ordered.py` | **READY** with minor cleanup — §B-05 (LRU dedup) and §B-06 (documentation) are polish items |
| Wireframe | `wireframe.py` | **READY** — exemplary file |
| Boozer analytic / radial | `boozer_analytic.py`, `boozer_fixed_state.py`, `boozer_radial_interp.py` | **READY** — strong code |
| Circular coil | `circular_coil.py` | **READY** with caveat — §B-07 (move spec params to data_fields) for autodiff completeness |
| Dipole | `dipole_field.py` | **READY** with caveat — §B-10 (origin-guard ordering) for AD safety |
| Curve geometry | `curve_geometry.py`, `framedcurve.py`, `finitebuild.py` | **NOT READY** — §C-01 layering must be addressed before this can be lifted out |
| Magnetic-axis iota | `magnetic_axis_helpers.py` | **NOT READY** — §B-01 (while_loop AD) and §B-04 (eig host-callback) both block production |
| Composition | `magneticfield_composition.py`, `mhd_reductions.py`, `objectives_flux.py`, `reductions.py` | **READY** — clean pass-through helpers |
| Permanent-magnet optimization | `pm_optimization.py` | **READY** — long but consistent |
| Regular-grid interpolant | `regular_grid_interp.py`, `interpolated_boozer_field.py`, `interpolated_field.py`, `surface_classifier.py` | **NOT READY** — §B-03 (mutable dict inside frozen dataclass) is a real IMMUTABLE-rule violation |
| Sampling / sharding | `sampling.py`, `sharding.py` | **READY** — modern API, good PRNG hygiene |
| Specs | `specs.py` | **READY** with caveat — §C-01 (cross-package lazy imports). Sound discriminant pattern, good pytree registration |
| Surfaces | `surface_fourier.py`, `surface_henneberg.py`, `surface_rzfourier.py`, `surface_classifier.py` | **READY** with caveat — `surface_fourier.py` and `surface_henneberg.py` are thin wrappers over `..geo.surface_fourier_jax`; if §C-01 is fixed, these become first-class JAX modules |
| Tracing | `tracing.py` | **NOT READY** — §B-01 (while_loop / fori_loop AD) blocks reverse-mode gradients through the trace drivers; document the contract clearly even if the fix is deferred |
| Scalar potential | `scalar_potential_rz.py` | **READY** with cleanup — §B-08 (remove `_TINY_SYMPY_TERM` once `_constant_like` is hardened) |

### Top three actions, in priority order

1. **Fix the layering violation (§C-01).** Move the curve and surface
   `*_pure` kernels into `jax_core/` so that `simsopt.geo` and
   `simsopt.field` import **down** from `jax_core/`. This is the
   blocker for the "JAX-only library" story and the largest single
   improvement to import-time hygiene.
2. **Document and (if possible) fix the AD-through-while_loop story
   (§B-01).** The `trace_*` and `on_axis_iota_rk` docstrings overstate
   gradient support. Either replace `while_loop` with bounded `scan`
   for the integrators that need reverse-mode (likely just
   `on_axis_iota_rk`) or document the contract explicitly so callers
   know to use `jax.jvp` and forward-mode only.
3. **Resolve the `dB` axis convention split (§B-02).** Either rename
   the divergent functions in `analytic_pure_fields.py` or transpose
   their outputs at the public boundary so a single
   "if you ask for `dB`, you get `[p, j, l]`" rule holds across the
   whole kernel layer.
