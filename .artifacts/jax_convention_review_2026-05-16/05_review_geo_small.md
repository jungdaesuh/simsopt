# JAX/simsopt Convention Audit — Small `simsopt/geo/*_jax*.py` Modules

Branch: `gpu-purity-stage2-20260405` at HEAD as of 2026-05-16
Audited files (eight): `_distance_jax.py`, `boozer_residual_jax.py`,
`curveobjectives_jax.py`, `framedcurve_jax.py`, `label_constraints_jax.py`,
`permanent_magnet_grid_jax.py`, `surface_fourier_jax.py`,
`surface_fourier_jax_cpu_ordered.py`.

## Executive summary

All eight files compile, are wired into the JAX-side import graph in
`simsopt/geo/__init__.py`, and uphold the M1/M3/M5 simsopt-port contract
documented in `CLAUDE.md`. Tensor-axis conventions, the stellsym scatter
pattern, the `nfp` quadrature cancellation, and the CPU-ordered Phase-2
parity twin are all faithful to their C++/CPU oracles. The cpu_ordered
twin in `surface_fourier_jax_cpu_ordered.py` is correctly gated and never
imported from the production fast path.

The findings cluster into three categories:

1. **Drift / SSOT violations** (medium severity): `surface_fourier_jax.py`
   carries a private divergent copy of `_as_jax_float64` /
   `_as_runtime_float64` that disables the GPU-residency contract held by
   the canonical `jax_core._math_utils` versions used elsewhere in the
   tree. `curveobjectives_jax.py` is largely a re-skin of CPU
   `curveobjectives.py` — its `CurveLengthJAX`, `LpCurveCurvatureJAX`,
   `LpCurveCurvatureBarrierJAX`, and `MeanSquaredCurvatureJAX` are
   structurally identical to the CPU originals.
2. **Public-API parity gaps**: `framedcurve_jax.py` ports `rotated_frame`,
   `rotated_frame_dash`, `frame_torsion`, `frame_binormal_curvature` but
   omits `frame_twist`, `rotated_frame_dcoeff_vjp`,
   `rotated_frame_dash_dcoeff_vjp`, and does not subclass `sopp.Curve`
   (documented choice). `LinkingNumberJAX.J()` returns a JAX float scalar
   while `LinkingNumber.J()` returns a Python int — a downstream type
   drift.
3. **JAX best-practice nits**: `framedcurve_jax.py` lacks `@jax.jit`
   wrappers on the per-input VJP bundles (the CPU sibling has them);
   `boozer_residual_jax.py` uses lazy-imports for three downstream JAX
   modules to avoid (real) circular dependencies; `_split_decision_vector`
   uses `int(x.shape[0])` and Python-time `ValueError`s.

No correctness bugs were found relative to the documented contracts.
The cpu_ordered twin's accumulation order and the stellsym scatter
indices match the C++ oracle operator-for-operator.

## Per-module findings

### A. `_distance_jax.py` (101 LOC) — point-cloud candidate cullers

#### Conventions

- **Privacy**: leading underscore in filename; **not** exported from
  `simsopt/geo/__init__.py` (`_BASE_CPU_GEO_MODULES`,
  `_ORDERED_JAX_CPU_GEO_BLOCK`, `_JAX_GEO_MODULES`,
  `_SIMSOPT_JAX_GEO_MODULES` confirm; verified). Sole consumers:
  `simsopt/geo/curveobjectives.py:708` and `:946` via inline
  `from ._distance_jax import ...`. Good.
- **Dtype contract**: `_stack_point_clouds` pins `np.float64` (lines
  16, 19). Inside JIT, `jnp.asarray(threshold, dtype=points.dtype)` at
  lines 51, 64 forces the threshold to float64. No silent promotion path.

#### JAX practices

- **JIT signatures**: `_within_collection_candidate_mask` uses
  `static_argnames=("num_base_curves",)` (line 40). `num_base_curves`
  feeds the triangular mask comparison `indices[None, :] < num_base_curves`
  — a Python int operand against a JAX array works fine; making it static
  guards against `num_base_curves` being a JAX tracer (which would force
  retrace per-value). Correct.
- **Padding semantics**: zero-padded points with `valid` masks; invalid
  pair distances replaced with `jnp.inf` before `jnp.min` (line 32, 65).
  No divide-by-zero risk because we never compute `1/dist`.
- **Boundary**: returns a Python `list[tuple[int, int]]`. `np.nonzero` at
  line 36 happens outside JIT (the JIT functions return masks; conversion
  is post-JIT). Good — keeps the public surface compatible with the C++
  candidate culler return type.

#### Severity

No findings.

#### Verdict

PASS. Clean, well-scoped private helper.

---

### B. `label_constraints_jax.py` (61 LOC) — Boozer inner-solve constraints

#### Conventions

- **Mirror of CPU `Volume.J()`/`Area.J()` semantics**: re-exports
  `volume_jax` / `area_jax` from `surface_fourier_jax` (lines 14-15) so
  the inner solve stays on-device. Confirmed against
  `surfaceobjectives.py:Volume.J()` and `:Area.J()` and the C++
  `surface.cpp:493,598` — no `nfp` multiplier, divisor is
  `nphi*ntheta` (`surface_fourier_jax.py:2417,2434`).
- **Toroidal flux convention**: `toroidal_flux_jax(A, gammadash2_at_phi, ntheta)`
  computes `sum(A * gammadash2_at_phi) / ntheta` (line 46). Matches
  `surfaceobjectives.py:ToroidalFlux.J()` at line 397: `np.sum(A * xtheta) / ntheta`.
- **G from currents**: `compute_G_from_currents(currents) = μ₀ * sum(|I_k|)`
  (line 60-61). The docstring formula `G = 2π * Σ|I_k| · μ₀/(2π)` reduces
  to this. Note: it uses `|I_k|` (signed currents are reduced to magnitudes)
  — this matches simsopt's Boozer convention but is worth a docstring
  note that signs are dropped.

#### JAX practices

- Pure `jnp` arithmetic; no host transfers. Fully traceable.
- No JIT decorator but compiles transparently inside callers.

#### Severity

- **Minor / docstring**: `compute_G_from_currents` drops current signs
  via `jnp.abs`. The docstring says `G = ... · μ₀/(2π)`, which omits the
  absolute-value. Surface this in the docstring so callers know the sign
  contract.

#### Verdict

PASS with minor docstring nit.

---

### C. `boozer_residual_jax.py` (828 LOC) — M1 + M3 Boozer residual

#### Conventions

- **M3 four-function contract**: `boozer_penalty_composed` (line 625),
  `boozer_penalty_grad_composed` (line 686),
  `boozer_residual_jacobian_composed` (line 743), and
  `boozer_residual_coil_vjp` (line 769) all present, all in `__all__`
  (lines 64-74). Signatures:
  - `boozer_penalty_composed(x, *, coil_arrays, quadpoints_phi, quadpoints_theta, mpol, ntor, nfp, stellsym, scatter_indices, optimize_G, weight_inv_modB=True, reduction_mode="default")` — kwargs-only.
  - `boozer_penalty_grad_composed(x, **kwargs)` — forwards to `value_and_grad(boozer_penalty_composed)`.
  - `boozer_residual_jacobian_composed(x, **kwargs)` — returns `(r, J)`.
  - `boozer_residual_coil_vjp(adjoint, *, gamma, xphi, xtheta, coil_arrays, iota, G, weight_inv_modB=False)` — different positional/kwargs signature because surface geometry is fixed.
  Internally consistent; cited by `tests/geo/test_boozer_derivatives_jax.py:37-40` (the M3 FD-validated suite).
- **Decision-vector packing**: `_split_decision_vector(x, optimize_G)` at
  line 87 unpacks `[sdofs, iota]` or `[sdofs, iota, G]`. The
  `int(x_jax.shape[0])` at line 90 and Python-time `ValueError` at
  line 91-95 mean dynamic shapes are not supported (acceptable — every
  caller knows its decision-vector size at trace time).
- **Scalar normalization**: `boozer_residual_scalar` divides by
  `num_res = 3*nphi*ntheta` (line 185). The module docstring at lines
  32-36 explains this matches the production CPU pipeline at
  `boozersurface.py:601-602` and that the raw `sopp.boozer_residual`
  C++ symbol does NOT carry this normalization. Aligns with CLAUDE.md
  conventions.
- **Weighted residual**: `_boozer_weighted_residual` at line 114 computes
  `r̃ = w · (G·B − |B|²·tang)` where `w = 1/|B|` (line 119-121) via
  `_explicit_rsqrt`. The custom-JVP `explicit_rsqrt` in
  `jax_core._math_utils.py:213` correctly produces non-finite tangents on
  `B² = 0`, which matches the documented "fail-closed" behaviour at line
  104-110.
- **VJP convention vs CLAUDE.md**: `boozer_residual_coil_vjp`'s signature
  takes `adjoint` first and (`gamma`, `xphi`, `xtheta`, `coil_arrays`,
  `iota`, `G`) as kwargs. This is the **raw** helper, distinct from the
  per-solve `res['vjp']` callback signature `(lm, booz_surf, iota, G)`
  documented in CLAUDE.md. The two surfaces are not in conflict — the
  helper is consumed by the operator-backed adjoint glue in
  `boozersurface_jax.py:25` (and tests).

#### JAX practices

- **Lazy imports** (`boozer_residual_jax.py:476-510`): three of them —
  `_get_surface_fns`, `_get_surface_xyzfourier_fns`, `_get_grouped_biot_savart`.
  These violate the user's "no dynamic imports" guardrail but are needed
  to break real circular dependencies:
  - `_get_surface_fns` / `_get_surface_xyzfourier_fns` defer to
    `simsopt.geo.surface_fourier_jax`, which transitively imports
    `simsopt.jax_core.__init__`, which (via `jax_core.surface_henneberg`
    and `jax_core.surface_fourier`) imports back into
    `simsopt.geo.surface_fourier_jax`. The bottom-of-file import workaround
    in `surface_fourier_jax.py:2766-2768` is the symmetric mitigation.
  - `_get_grouped_biot_savart` defers to `simsopt.field.biotsavart_jax`,
    which is a thin shim re-exporting from `simsopt.jax_core.biotsavart`.
    Inspecting the shim (`biotsavart_jax.py:1-55`), there is no transitive
    dependency on `simsopt.geo`; the lazy import here appears
    over-cautious — it could safely be lifted to module top level.
  - Severity: **low** for the surface helpers (genuine cycle break); **low/cosmetic**
    for the Biot-Savart helper (could be promoted to top-level import).
- **CPU-ordered LS scalar+grad** at lines 347-473
  (`boozer_residual_scalar_and_grad_cpu_ordered`): hand-rolled
  `lax.fori_loop` over `(phi, theta)` mirroring `sopp.boozer_residual_ds`.
  The implementation looks careful: scalar accumulators, per-point
  `(value, grad)` carry, separate iota/G gradient branches. One
  nitpick: line 374 builds zero via `jnp.sum(B, dtype=B.dtype) - jnp.sum(B, dtype=B.dtype)` — `jnp.zeros((), dtype=B.dtype)` is simpler and cheaper.
- **`jacfwd` choice** at line 765: `J = jax.jacfwd(_boozer_residual_vector_composed)(x, **kwargs)`. For Boozer the residual has
  `n_res = 3·nphi·ntheta` and decision size `~nsurfdofs+2` typically
  with `nsurfdofs ≪ n_res`. With output-dimension ≫ input-dimension,
  **`jacrev` (reverse-mode)** would be cheaper. `jacfwd` is mathematically
  correct, but suboptimal for this pipeline.
- **Surface-DOF zero gradient wrappers**: `boozer_residual_grad` /
  `boozer_residual_hessian` (lines 228-312) build a packed
  `(nsurfdofs+2)`-vector with zero surface entries and `jax.grad` /
  `jax.hessian` through `_boozer_objective_from_packed`. This wastes
  autodiff work on a block that is provably zero (since B, xphi, xtheta
  are constants). The docstrings note the limitation. Callers needing
  the surface block must route through the M3 composed pipeline. Fine
  by intent but inefficient by autodiff.

#### Severity findings

- **M-1** `boozer_residual_jacobian_composed` (line 743-766) uses
  `jax.jacfwd`. For typical Boozer cases (`n_res ≫ n_x`), `jacrev` is
  cheaper. Severity **medium** (performance-only).
- **L-1** Lazy import of `grouped_biot_savart_B` (line 506-510) is
  unnecessary; the shim it loads has no circular dependency.
  Severity **low** (cosmetic).
- **L-2** Cheap zero pattern at line 374 uses `sum(B)-sum(B)`. Replace
  with `jnp.zeros((), dtype=B.dtype)`. Severity **low**.
- **L-3** `compute_G_from_currents` uses `|I_k|`; document the
  sign-dropping in `label_constraints_jax.py:60-61` docstring.
  Severity **low** (docstring).

#### Verdict

PASS. M3 contract complete; one performance fix (`jacfwd→jacrev`) and
a cosmetic lazy-import cleanup recommended.

---

### D. `surface_fourier_jax.py` (2768 LOC) — SurfaceXYZTensorFourier and SurfaceXYZFourier pure-JAX kernels

#### Conventions

- **Stellsym scatter indices** (`stellsym_scatter_indices`, line 1154-1179):
  - x uses cos-cos + sin-sin (`_is_stellsym_xy`, line 1128-1138).
  - y and z use cos-sin + sin-cos (`_is_stellsym_z`, line 1141-1151).
  Cross-checked against C++ `SurfaceXYZTensorFourier::skip` at
  `src/simsoptpp/surfacexyztensorfourier.h:1233-1242`:
  - dim 0 (x) skip if `(n≤ntor && m>mpol) || (n>ntor && m≤mpol)` → keep
    cos-cos and sin-sin. Matches `_is_stellsym_xy`.
  - dim 1 (y) skip if `(n≤ntor && m≤mpol) || (n>ntor && m>mpol)` → keep
    cos-sin and sin-cos. Matches `_is_stellsym_z`.
  - dim 2 (z) same as dim 1. Matches.
  Stellsym DOF ordering is correct per CLAUDE.md.
- **Tensor convention**: surface coefficient Jacobians (line 2503, 2513,
  2523) return `(nphi, ntheta, 3, ndofs)` where the last axis is the DOF
  index, third axis is the Cartesian component. Compatible with sopp.
- **Volume / area normalization**: `surface_volume` (line 2399) returns
  `Σ γ·n / (3·nphi·ntheta)`; `surface_area` (line 2420) returns
  `Σ |n| / (nphi·ntheta)`. Both omit `nfp`. CPU `Surface::volume()` at
  `src/simsoptpp/surface.cpp:598` and `Surface::area()` at line 493
  follow the same pattern. The CLAUDE.md note "nfp factor cancels with
  quadrature step `1/(nfp*nphi)`" is upheld implicitly because the
  quadrature is parameterized over `[0, 1/nfp)` with `nphi` samples in
  the JAX kernels and the Python `1/(nphi*ntheta)` normalizer matches
  the half-period integral.
- **BC enforcer / `clamped_dims`** (`_bc_enforcer_grid` line 353): the
  enforcer `E(phi, theta) = sin(nfp·phi/2)² + sin(theta/2)²` is applied
  multiplicatively to the cos-cos coefficient block per
  `surfacexyztensorfourier.h:903-913`. Implemented as
  `hat + block_hat * (E - 1)` (line 460-467, 491-499). Direct C++
  parity surface.
- **Basis convention**: `build_theta_basis` (line 172) and
  `build_phi_basis` (line 239) construct `(W, dW)` with `dW` being the
  derivative w.r.t. quadpoint parameters (not raw angles), absorbing the
  `2π` chain-rule factor. Matches the production fast path; differs
  from `surface_fourier_jax_cpu_ordered.py` which keeps `2π` external
  for C++ parity.

#### JAX practices

- **`jacfwd` choice for surface coefficient Jacobians**: `dgamma_by_dcoeff` and friends
  (lines 2496-2556) use `jax.jacfwd`. The output shape is `(nphi, ntheta, 3, ndofs)`
  → flat size `~3·nphi·ntheta`. Decision-vector size is `ndofs`. Typically
  `3·nphi·ntheta > ndofs`, so reverse-mode `jacrev` would build fewer
  cotangent paths. The choice here is correct per CLAUDE.md which lists
  these as `jax.jacfwd`, but the asymptotic justification is reversed —
  jacfwd is preferred when `n_in ≤ n_out` only if it parallelizes well
  on accelerators. For surface coefficient Jacobians the autodiff cost
  scales linearly with `n_in = ndofs`, so jacfwd wins only when
  `ndofs < 3·nphi·ntheta`. For typical stage-1 fixed-boundary problems
  with `ndofs ≈ O(100)` and `nphi·ntheta ≈ O(1000)`, jacfwd is the right
  call. Confirmed.
- **`vmap` for `_lin` paired-point evaluators** when `clamped_dims` is
  active (e.g. `surface_gammadash1_lin` line 626-643): falls back to
  `jax.vmap(jax.jacfwd(_eval_single))(qp)` because the multiplicative
  clamped correction interacts with the angle product rule in a way the
  analytic formula doesn't cover. Pragmatic and correct.
- **`lax.scatter` for DOF unpacking** (`dofs_to_xyzc`, line 1225-1233):
  scatter-set with `PROMISE_IN_BOUNDS`, sorted, unique indices. Good
  JAX practice for the fixed-shape DOF→matrix mapping.
- **End-of-file deferred import** (lines 2763-2768): the bottom-of-file
  `from simsopt.jax_core._math_utils import unit_vector_axis_last as _unit_vector_axis_last`
  is a workaround for a real circular import between `simsopt.geo` and
  `simsopt.jax_core`. The pattern works because `_unitnormal` (line 1114)
  is never called at module-load time; by the time any caller invokes
  it, the import has resolved. Comment explains the rationale. Acceptable
  but fragile — any new top-of-file import of `_unit_vector_axis_last`
  would break.

#### Severity findings

- **M-2** Divergent `_as_jax_float64` / `_as_runtime_float64`
  (lines 100-108): `_as_runtime_float64` does `del reference` and
  delegates to `_as_jax_float64`. The canonical
  `simsopt.jax_core._math_utils.as_runtime_float64` (line 135) uses the
  reference's dtype/device for GPU residency. By bypassing the reference,
  this module loses GPU-residency preservation when mixed with GPU
  buffers. Severity **medium** (SSOT violation; explicit GPU contract
  drift). Recommend: import `as_jax_float64` / `as_runtime_float64`
  from `jax_core._math_utils` and remove the local copies.
- **L-4** `_as_runtime_float64` consumes a keyword-only `reference`
  argument it then immediately deletes (line 107). At minimum, the
  reference is preserved at call sites and could be honored to keep
  GPU residency. Same root cause as M-2.

#### Verdict

PASS with one SSOT/GPU-residency drift (M-2). All core JAX kernels
correct; stellsym scatter and clamped-dim enforcer match C++ oracle.

---

### E. `surface_fourier_jax_cpu_ordered.py` (721 LOC) — Phase-2 parity twin

#### Conventions and parity contract

- **Scope and gating**: module docstring (lines 1-24) is explicit:
  - Diagnostic-grade only; production fast paths in
    `surface_fourier_jax.py` are unchanged.
  - "Only the parity backend
    (`SIMSOPT_BACKEND_MODE=jax_cpu_parity`/`jax_gpu_parity` →
    `simsopt.backend.is_parity_mode`) routes through them."
  Verified that no production caller imports from this module
  (`grep` over `src/simsopt/`) confirms only test harnesses and parity
  ladders consume `surface_*_cpu_ordered` symbols.
- **Loop ordering** (line 166-173): `m → n → ...` matches the C++
  `gamma_impl` ordering (`surfacexyztensorfourier.h:127`). The
  `lax.fori_loop` uses scalar accumulators, ensuring per-point sum
  order matches.
- **`2π` chain rule**: the dash kernels apply the `2π` factor *outside*
  the rotation (lines 235, 265), matching the docstring claim and the
  C++ accumulation order. The basis caches contain derivatives w.r.t.
  raw angle (φ, θ) — not w.r.t. quadpoint parameters — exactly as
  noted in the docstring at lines 68-71 and 94-95.
- **`basis_fun_phi[_dash]` correctness** (lines 73-90): cross-checked
  against C++ `basis_fun_phi` / `basis_fun_phi_dash` at
  `surfacexyztensorfourier.h:1177-1189`:
  - For `n ≤ ntor`: cos block uses `cos(nfp·n·phi)`, dash uses
    `-nfp·n·sin(nfp·n·phi)`. JAX `arg_cos = nfp · n_cos_idx · phi`,
    `cos_block_dash = -(nfp·n_cos_idx) · sin(arg_cos)`. Match.
  - For `n > ntor`: sin block uses `sin(nfp·(n-ntor)·phi)`, dash uses
    `nfp·(n-ntor)·cos(...)`. JAX uses `n_sin_idx = arange(1, ntor+1)`,
    `sin_block_dash = (nfp·n_sin_idx) · cos(arg_sin)`. Match.
- **`_stellsym_skip_mask` and `_coeff_dof_index_table`** (lines 400-457):
  Python-side computation of which (d, m, n) tuples are free DOFs and
  their DOF index in the C++ counter walk. Matches `skip(d, m, n)` at
  `surfacexyztensorfourier.h:1233`. The `counters` array is then used
  to scatter per-(d, m, n) basis-product planes into a dense
  `(nphi, ntheta, 3, ndofs)` Jacobian.
- **Dtype**: `cache_phi`/`cache_theta` flow through `_as_jax_float64`
  (lines 73, 95) → all derived arrays inherit `float64`. `out_zero`
  uses `dtype=cache_phi.dtype` (line 504). No silent float32 path.

#### JAX practices

- **`@partial` + `jax.vmap` over k1/k2** (lines 287-296, 326-339,
  370-383): vmap composition over the quadrature grid; each
  scalar kernel cell evaluates the inner `lax.fori_loop` over basis
  modes. Idiomatic for the parity contract — preserves cross-grid
  determinism while staying on-device.
- **Dense `_dgamma_by_dcoeff_dense`** (lines 460-573): builds the
  Jacobian by iterating `(d, m, n)` on the host and emitting
  `out.at[:, :, dim, counter].set(plane)` per DOF. For
  `(2·mpol+1)·(2·ntor+1)·3` ≈ thousands of writes, this constructs a
  long XLA scatter chain. Acceptable for diagnostic parity (correctness
  over throughput); not used in production hot paths.

#### Severity findings

None — the module is explicitly diagnostic, the C++ parity oracle is
matched operator-for-operator, and the production fast path is
unaffected.

#### Verdict

PASS. Excellent parity twin; gating contract documented and respected.

---

### F. `framedcurve_jax.py` (599 LOC) — JAX-backed framed curve wrappers

#### Conventions

- **Class structure** (lines 312-599): `FrameRotationJAX`,
  `ZeroRotationJAX`, `FramedCurveFrenetJAX`, `FramedCurveCentroidJAX`,
  and a base `_FramedCurveJAXBase`. All four classes subclass
  `Optimizable` (and only `Optimizable` — **not** `sopp.Curve`). The
  CPU sibling `FramedCurve` at `framedcurve.py:41` subclasses both
  `sopp.Curve, Curve`. The JAX docstring at lines 13-16 acknowledges
  the deliberate divergence: "The wrappers do **not** subclass
  `sopp.Curve` — they sit alongside the C++/CPU framed-curve classes
  as a parallel JAX implementation." Acceptable.
- **DOF graph**: `_FramedCurveJAXBase.__init__` declares
  `depends_on=[curve, rotation]` (line 424-425) and registers an empty
  DOF vector (`x0=np.asarray([])`). Matches the CPU contract where
  `FramedCurve` holds no DOFs of its own (DOFs live in `curve` and
  `rotation`).
- **`FrameRotationJAX.__init__`** (lines 323-338): order, scale, dofs
  precomputed `jac`/`jacdash` matrices. Same signature as
  `FrameRotation.__init__` at `framedcurve.py:428`.
- **`alpha` / `alphadash` VJPs**: `FrameRotationJAX.dalpha_by_dcoeff_vjp`
  (line 371-378) builds the gradient with a NumPy matmul through
  `self.jac.T`, returning `Derivative({self: gradient})`. CPU uses
  `sopp.vjp(v, self.jac)`. Result is identical (both are
  `J^T v * scale`). Acceptable.
- **Public method coverage** vs CPU:
  - **Present** in JAX: `rotated_frame`, `rotated_frame_dash`,
    `frame_torsion`, `frame_binormal_curvature`,
    `dframe_torsion_by_dcoeff_vjp`,
    `dframe_binormal_curvature_by_dcoeff_vjp`.
  - **Missing** from JAX (present in CPU `FramedCurve`):
    `frame_twist`, `dframe_twist_by_dcoeff_vjp`,
    `rotated_frame_dcoeff_vjp`, `rotated_frame_dash_dcoeff_vjp`.
  This is a real public-API parity gap. If downstream code calls
  `framed_curve.frame_twist()` on a JAX-backed framed curve, it will
  `AttributeError`. Severity depends on whether `frame_twist` is part
  of the M5 / single-stage path.

#### JAX practices

- **Per-input VJPs without `@jax.jit`** (lines 136-310): six 4-/5-/6-arg
  VJP bundles, each invoking `jax.vjp(...)[1](v)[0]` five or six times.
  The CPU sibling at `framedcurve.py:667-707` wraps these in `@jit`
  decorators. The JAX-side wrapper does **not** — every call retraces.
  Severity **medium** for hot paths.
- **`@property quadpoints`** (line 427-429): JAX `_FramedCurveJAXBase`
  exposes `quadpoints` as a derived property of `self.curve.quadpoints`.
  CPU sets `self.quadpoints` via `sopp.Curve.__init__`. Read paths
  match; assignment (`framed.quadpoints = ...`) would `AttributeError`
  on JAX side. Unlikely in practice.
- **`ZeroRotationJAX` precompute** (line 397): `self._zero =
  jnp.zeros(int(quad_array.size), ...)`. The CPU `ZeroRotation` does
  the same precomputation pattern (`framedcurve.py:485`). Both ignore
  the `quadpoints` argument in `alpha(quadpoints)` / `alphadash(quadpoints)`.
  This is a latent bug in **both** classes (constructor `quadpoints`
  size pins the output, runtime call argument is silently discarded),
  but it's mirrored faithfully so no JAX-side regression.

#### Severity findings

- **M-3** Missing `frame_twist`, `dframe_twist_by_dcoeff_vjp`,
  `rotated_frame_dcoeff_vjp`, `rotated_frame_dash_dcoeff_vjp`. The
  `frame_twist` is referenced by upstream code (e.g.,
  `simsopt.geo.framedcurve.FramedCurve.frame_twist` at `framedcurve.py:66`).
  Downstream code that calls into a polymorphic `framed_curve`
  reference will fail on JAX wrappers. Severity **medium**.
- **L-5** Per-input VJP bundles in `framedcurve_jax.py:136-310` lack
  `@jax.jit`. CPU sibling decorates them. Adds compile overhead per
  call. Severity **low** (performance, not correctness).

#### Verdict

PASS with caveats. Functionally correct for the documented surface;
public-API parity gap on `frame_twist` etc. should be noted in the
M-level acceptance contract or filled in.

---

### G. `curveobjectives_jax.py` (394 LOC) — Optimizable wrappers for curve objectives

#### Conventions

- **Imports from CPU `curveobjectives`** (lines 16-39): pulls
  `curve_length_pure`, `Lp_curvature_pure`, `_curve_length_grad`,
  `_curve_msc_grad`, `cc_distance_pure`, `cc_distance_barrier_pure`,
  `cs_distance_pure`, `_cc_distance_grad`, etc. — i.e., the JAX
  wrappers consume the CPU module's pure JAX functions.
- **`CurveLengthJAX`, `LpCurveCurvatureJAX`, `LpCurveCurvatureBarrierJAX`,
  `MeanSquaredCurvatureJAX`**: structurally identical to their CPU
  counterparts. CPU `CurveLength.J()` at `curveobjectives.py:170-185`
  already routes through `curve_length_pure` and
  `_curve_length_grad`. The JAX wrapper at lines 60-70 does the same.
  Likewise for `LpCurveCurvature` and friends. Effectively duplicates
  the CPU classes; the only contractual difference is that the JAX
  classes carry the `JAX` suffix for explicit lane disambiguation.
- **`CurveCurveDistanceJAX` / `CurveCurveDistanceBarrierJAX` /
  `CurveSurfaceDistanceJAX`**: these are the **real** wrappers — they
  bypass the C++ candidate culler in favor of all-pairs JAX evaluation
  (docstrings at lines 4-7, 194, 234, 269). Useful when the JAX backend
  is forced.
- **`LinkingNumberJAX.J()`** (line 365-388): returns `total` accumulated
  as `_as_jax_float64`. CPU `LinkingNumber.J()` at
  `curveobjectives.py:1231-1266` casts each `pair_linking_number_pure`
  contribution to `int(...)` and accumulates **Python ints**. Result
  type drift: CPU returns a Python int (or float, depending on path),
  JAX returns a `jax.Array` float64 scalar. Downstream callers that
  type-check `isinstance(j, int)` may misbehave. Note: CPU also has a
  JAX-backend branch at line 1232-1260 that mirrors the JAX wrapper
  exactly but ultimately calls `int(contribution)` before summing. The
  JAX wrapper does not perform that final cast.
- **`_iter_curve_pair_indices`** (lines 171-174): matches CPU at
  `curveobjectives.py:606-609` operator-for-operator.

#### JAX practices

- **`zip(..., strict=True)`** at line 37 of `_distance_jax.py` — Python
  3.11 syntax used correctly elsewhere. `curveobjectives_jax.py` uses
  `zip(curve_positions, curve_tangents)` (line 311, 329) without
  `strict` — minor inconsistency.
- **No `@jit` decorators on the wrapper methods** — the wrappers
  themselves are Python; the underlying pure functions (imported from
  CPU `curveobjectives`) are `@jit`-decorated. Acceptable: the wrapper
  is the boundary between SIMSOPT's Optimizable graph and JAX tracing.

#### Severity findings

- **M-4** `LinkingNumberJAX.J()` returns `jax.Array` scalar; CPU
  `LinkingNumber.J()` returns a Python int. Type drift across the
  CPU/JAX boundary. Severity **medium** (silent type change at a
  public API boundary).
- **L-6** `CurveLengthJAX`, `LpCurveCurvatureJAX`,
  `LpCurveCurvatureBarrierJAX`, `MeanSquaredCurvatureJAX` are
  structurally identical to their CPU siblings. Severity **low**
  (SSOT-by-duplication; could be a deliberate explicit-lane choice
  but worth documenting in the M-level contract).
- **L-7** `_iter_curve_pair_indices` in `_CurveCurveDistanceJAXBase`
  duplicates CPU `_iter_curve_pair_indices`. Same code, two homes.

#### Verdict

PASS with type-drift caveat on `LinkingNumberJAX.J()`.

---

### H. `permanent_magnet_grid_jax.py` (226 LOC) — Permanent-magnet JAX payload

#### Conventions

- **Scope difference vs CPU**: CPU `PermanentMagnetGrid` at
  `permanent_magnet_grid.py:14` is a heavyweight class handling grid
  initialization, FAMUS file reading, plasma-boundary projection, etc.
  The JAX `PermanentMagnetGridJAX` is a **frozen dataclass** containing
  the **fixed-state payload** (precomputed matrices, dipole grid, m_maxima)
  needed for on-device optimization. Not an alias; a deliberate scope
  reduction. The docstring at lines 30-35 ("Immutable JAX payload for a
  fixed permanent-magnet optimization state") is explicit.
- **`from_cpu` constructor** (line 56-86): stages a CPU
  `PermanentMagnetGrid` into the JAX payload. Uses `_as_jax_float64`
  for all arrays. Reads `pm_grid.plasma_boundary.nfp/stellsym` for the
  meta fields.
- **`from_fixed_state` constructor** (line 88-185): builds the payload
  from explicit arrays. Calls `dipole_field_Bn` (from
  `jax_core.dipole_field`) to compute the dipole field matrix A_obj.
  Standalone path that doesn't require a CPU `PermanentMagnetGrid` instance.
- **MwPGP step-size rule** (`mwpgp_alpha_from_grid`, lines 220-226):
  returns `2*(1-1e-5) / ATA_scale`. Mirrors the C++ MwPGP convention
  for the safety factor.

#### JAX practices

- **`@dataclass(frozen=True)`** (line 33): immutable container.
- **`jax.tree_util.register_dataclass`** (lines 188-211): registers the
  dataclass as a pytree with explicit `data_fields` / `meta_fields`.
  `pol_vectors: jax.Array | None = None` is registered as a data field
  (line 200). Implication: when `pol_vectors` is `None`, the pytree
  structure differs from when it's an array. JIT caches the structure,
  so `None`-versus-array calls will trigger separate compilations.
  Severity: **low** — expected behaviour when an optional pytree node
  is conditionally populated.
- **`_reshape_moments`** (lines 24-30): defensive validation of moment
  array shapes. Coerces `(ndipoles*3,)` to `(ndipoles, 3)` and rejects
  other shapes. Good — surfaces shape errors at staging time, not deep
  in the optimizer.
- **`jnp.linalg.svd`** at line 147: extracts σ₁ for `ATA_scale = σ₁²`.
  This is the production cost computation. Acceptable.
- **Type hints**: arguments use `object` for some inputs (e.g., line
  92-104). Not strictly wrong, but losing type information. Could be
  `np.ndarray | jax.Array`.

#### Severity findings

- **L-8** Optional pytree field `pol_vectors: jax.Array | None`
  triggers JIT-cache structural drift between `None`-bearing and
  array-bearing instances. Document in the dataclass docstring so
  callers know to keep `pol_vectors` consistently typed across calls.

#### Verdict

PASS. Clean fixed-state payload; pytree registration done correctly.

---

## Cross-cutting positive notes

- **All eight files pin float64**: every `_as_jax_float64` /
  `_as_runtime_float64` / `jnp.float64` use sites pass float64 explicitly.
  No implicit float32 promotion path found.
- **No `if traced:` or tracer-introspection patterns**. JAX hygiene is
  clean.
- **CLAUDE.md M3 four-function contract** is upheld:
  `boozer_penalty_composed`, `boozer_penalty_grad_composed`,
  `boozer_residual_jacobian_composed`, `boozer_residual_coil_vjp` all
  present in `__all__` of `boozer_residual_jax.py` with consistent
  signatures.
- **Stellsym scatter** matches the C++ `skip(d, m, n)` predicate
  operator-for-operator: x = cos-cos+sin-sin; y = cos-sin+sin-cos;
  z = cos-sin+sin-cos. CLAUDE.md convention preserved.
- **cpu_ordered twin** correctly gated: no production caller imports
  it; only parity tests do.
- **`_distance_jax.py`** is a real private helper — not re-exported
  from `simsopt/geo/__init__.py`; only consumed from
  `simsopt/geo/curveobjectives.py` via inline imports.

## Per-file verdict matrix

| File | Verdict | Severity highlights |
|------|---------|---------------------|
| `_distance_jax.py` | PASS | none |
| `boozer_residual_jax.py` | PASS | M-1 (`jacfwd→jacrev` perf), L-1/L-2/L-3 |
| `curveobjectives_jax.py` | PASS | M-4 (`LinkingNumberJAX` type drift), L-6/L-7 |
| `framedcurve_jax.py` | PASS | M-3 (missing `frame_twist` etc.), L-5 |
| `label_constraints_jax.py` | PASS | minor docstring (sign of currents) |
| `permanent_magnet_grid_jax.py` | PASS | L-8 (optional pytree node) |
| `surface_fourier_jax.py` | PASS | M-2 (SSOT/`_as_runtime_float64` drift), L-4 |
| `surface_fourier_jax_cpu_ordered.py` | PASS | none |

Severity legend: **M** = medium (worth addressing in next port pass);
**L** = low / cosmetic / docstring.

## Aggregate recommendation

Eight files passing, no correctness bugs. Two medium-severity
follow-ups are worth filing as issues:
1. M-2: consolidate `_as_runtime_float64` SSOT to honor GPU residency.
2. M-3: complete the `framedcurve_jax` public surface (`frame_twist`,
   `rotated_frame_dcoeff_vjp`, `rotated_frame_dash_dcoeff_vjp`).

Other findings are performance or documentation nits.
