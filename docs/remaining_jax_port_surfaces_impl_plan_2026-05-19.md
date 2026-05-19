# Remaining JAX-Port Surfaces — Implementation Plan (2026-05-19)

- **Branch:** `gpu-purity-stage2-20260405`
- **HEAD reviewed:** `175a04323` (live tree on 2026-05-19)
- **Author intent:** harden the JAX port over the *remaining* unported or
  not-fully-claimed differentiable / math surfaces called out by the
  port-gap audit. No toy lanes. Research-prod-grade only.
- **Authoritative reference contracts:**
  - `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`
    (parity ladder SSOT — `direct_kernel`, `relaxed_kernel`,
    `derivative_heavy`, `direct_hessian_oracle`,
    `ls_wrapper_gradient`, `exact_well_conditioned_adjoint`,
    `exact_ill_conditioned_adjoint`, `branch_stable_resolve`,
    `fd_gradient`, `gpu_runtime`, `reduction_cpu_gpu`).
  - `docs/parity_dual_mode_contract_2026-05-08.md` (mode matrix:
    `native_cpu`, `jax_cpu_parity`, `jax_cpu_fast`, `jax_gpu_parity`,
    `jax_gpu_fast`, `jax_mps_smoke`).
  - `src/simsopt/backend/runtime.py` for `BackendPolicy` knobs.
- **Official API references checked 2026-05-19:**
  - JAX 0.10 / jaxlib 0.10 on CUDA 12 for this repo lane (SM ≥ 5.2;
    Linux driver ≥ 525). Current upstream JAX docs recommend CUDA 13 for
    new installs, but this repo's release proof remains pinned to the
    CUDA 12 `jax==0.10.0` / `jaxlib==0.10.0` wheel set.
  - `jax.scipy.optimize.minimize` (BFGS only); `jax.scipy.linalg`;
    `jax.lax.while_loop` / `jax.lax.scan`; public `jax.shard_map`;
    `jax.custom_vjp` for reverse-mode implicit differentiation.
  - `optimistix` (optional `simsopt[JAX_OPTIMISTIX]` lane: LM,
    IndirectLM, Dogleg, BFGS) for least-squares / minimization;
    augmented-Lagrangian wrapper for equality-constrained problems.
  - SciPy spline APIs (`InterpolatedUnivariateSpline`,
    `RectBivariateSpline`) for extracting and replaying FITPACK spline
    coefficients through `get_knots()` / `get_coeffs()`.

---

## 0. Why This Plan Exists

The port-gap audit identified the following residual surfaces still
backed by `numpy` / `scipy` (and in some cases mutable host state),
either because they were intentionally deferred (low priority for the
Stage-2 / single-stage critical path) or because they require a
non-trivial design decision (state freezing, host vs. device adjoint
boundary, MPI replacement). They are:

| # | Surface | Audit classification |
|---|---|---|
| 1 | `src/simsopt/mhd/bootstrap.py:27` `compute_trapped_fraction` | pure NumPy physics formula; differentiable array math, not JAX-ported |
| 2 | `src/simsopt/mhd/bootstrap.py:173` `j_dot_B_Redl` | formula-heavy Redl bootstrap calculation; differentiable-ish dense math, not JAX-ported |
| 3 | `src/simsopt/mhd/vmec_diagnostics.py:1208` `vmec_compute_geometry` | real geometry math, but currently built around VMEC/SciPy spline state — portable only after freezing spline coefficients |
| 4 | `src/simsopt/mhd/vmec_diagnostics.py:1770` `vmec_fieldlines` | coordinate-line geometry path; uses SciPy `newton` for `theta_pest -> theta_vmec`, not an ODE integrator |
| 5 | `src/simsopt/mhd/profiles.py` | `ProfilePolynomial`, `ProfileSpline`, `ProfileScaled`, `ProfilePressure` are not JAX-native |
| 6 | `src/simsopt/geo/qfmsurface.py` `QfmSurface` solver orchestration | SciPy `minimize` loop and `surface.x = x` mutation are not ported; the differentiable residual kernel is ported as `QfmResidualJAX` |
| 7 | `src/simsopt/solve/serial.py` and `src/simsopt/solve/mpi.py` | generic Optimizable SciPy/MPI wrappers; host orchestration over arbitrary mutable graphs |
| 8 | Live PM / wireframe workflows | fixed-state kernels and JAX wrappers exist, but live mutable workflows, FAMUS/plot/VTK writers, pruning / final-adjustment are not fully JAX-native |

This plan turns each row into a milestone with concrete code edits,
verification contract, and an acceptance gate that hooks into the
existing parity-ladder lane definitions. Nothing in this document
loosens the strict CPU↔JAX byte-identity gate
(`_pre_newton_census_gate_failures`) or invents a new tolerance lane.

---

## 1. Goals and Non-Goals

### 1.1 Goals

- Ship JAX-native implementations of every formula-heavy surface above
  with explicit autodiff support on both `cpu` and `cuda` platforms:
  forward- and reverse-mode for pure algebraic kernels, and reverse-mode
  implicit differentiation for converged root/solver boundaries,
  end-to-end traceable from public Optimizable boundaries down to pure
  JAX kernels in `simsopt.jax_core`.
- Preserve the established public API on the import side
  (`simsopt.mhd.bootstrap.compute_trapped_fraction`, etc.) — the JAX
  variants live under a `*_jax` namespace and are toggled by the
  `SIMSOPT_BACKEND_MODE` SSOT (`SIMSOPT_BACKEND` remains a legacy
  compatibility selector only).
- Eliminate every load-bearing `simsoptpp` import along the new JAX
  paths. `simsopt.jax_core.*` must remain `simsoptpp`-free.
- Drive validation through the existing parity-ladder contract: each
  surface gets at least one independent oracle (C++ symbol where
  available, closed form, finite difference, or pinned reference
  dataset).
- Hold the `jax_*_parity` modes to the byte-identity gate; keep
  `jax_*_fast` modes opt-in researcher lanes that fail the same gate
  by construction.

### 1.2 Non-Goals

- We do not rewrite VMEC, SPEC, or upstream Fortran/C++ binaries.
- We do not rewrite `mpi4py`-based multi-host orchestration. The MPI
  worker pool stays. Only the inner finite-difference Jacobian
  assembly gets a shard-map alternative for the JAX-aware problem
  class.
- We do not promise byte-identity output writers (FAMUS, VTK,
  matplotlib). These remain host artefacts.
- We do not promise that legacy `metal` paths come back — `jax-metal`
  is unmaintained; Apple-GPU smoke continues to ride `jax-mps`.
- We do not introduce an SLSQP-equivalent JAX solver. Equality-constrained
  JAX paths use an augmented-Lagrangian wrapper; the existing host SLSQP
  path remains the `native_cpu` reference path and is never selected
  automatically after a JAX solver fails.

### 1.3 Definition of Done (per milestone)

A milestone is *Done* only when **all** of the following hold:

1. Public Python API exposes a JAX entrypoint that is import-safe with
   `JAX_PLATFORM_NAME=cpu`, `JAX_PLATFORM_NAME=cuda`, and under the
   mode SSOT.
2. `simsopt.jax_core` retains its `simsoptpp`-free invariant.
3. Parity tests cite independent oracles per `tests/REVIEWER_ORACLE_LINT.md`.
4. New code carries `jax.transfer_guard("disallow")`-clean call paths
   for the hot inner loops (one host materialization per outer step at
   most, gated by the public wrapper boundary).
5. Lint (`ruff check`, `ruff format`), `mypy` (no new regressions on
   touched files), and the regression suite in `CLAUDE.md` Validation
   block all pass for both the public pure-JAX lane and the private
   `private_optimizer_runtime` lane (where applicable).

---

## 2. Cross-Cutting Design Decisions

These apply to every milestone below.

### 2.1 State-freezing rule

When SciPy splines or VMEC wout-state appear in the source path, the
JAX adapter must accept a **frozen state pytree** (knots + coefficients
or full-grid arrays) rather than holding a live mutable handle. Pattern:

```python
@dataclasses.dataclass(frozen=True)
class VmecFrozenSplineState:
    s_full_grid: jax.Array         # (ns,)
    s_half_grid: jax.Array         # (ns-1,)
    rmnc: jax.Array                # (mnmax, ns)
    zmns: jax.Array                # (mnmax, ns)
    lmns: jax.Array                # (mnmax, ns-1)   half-grid
    # ... and so on
    xm: jax.Array
    xn: jax.Array
    stellsym: bool                  # static / Pythonic
    mnmax: int                      # static / Pythonic
    mnmax_nyq: int                  # static / Pythonic
```

A host helper `vmec_freeze_splines(vmec_or_splines)` materializes the state once
(from the existing `vmec_splines(vmec)` output) so the JAX path never
re-enters Python-level spline classes. Static dimensions are tracked
as pytree auxiliary data via `jax.tree_util` registration or plain
Pythonic ints, as we already do in `surface_rzfourier.py`; do not add
Equinox to the base JAX runtime for this state carrier.

### 2.2 Radial interpolation kernel

`InterpolatedUnivariateSpline` with `k=3` is the upstream default for
VMEC radial splines, and `RectBivariateSpline` / `interp1d(kind="cubic")`
appear in the trapped-fraction extrema path. The JAX replacement must
replay the same fitted spline representation where the host API exposes
it; do not substitute Catmull-Rom or a natural cubic spline under the
same parity claim. Implementation:

- Reuse the SciPy fit on the host (one-shot), but extract coefficient
  arrays and knot vectors via `get_knots()` / `get_coeffs()` and ship
  them through `VmecFrozenSplineState`.
- On-device evaluation via stable FITPACK-compatible B-spline de Boor
  routines in `simsopt.jax_core/_spline_utils.py` (new file):
  `bspline_eval_1d(knots, coeffs, degree, s)`,
  `bspline_deriv_1d(...)`, and a tensor-product 2-D evaluator for
  `RectBivariateSpline` coefficients. Use `jax.vmap` over the query
  axis and over the `mnmax` / `mnmax_nyq` axes.
- At-knot parity is the hard first gate; off-knot parity must be
  compared directly against the SciPy object and recorded as a measured
  interpolation replay tolerance, not described as byte-identical until
  proven.

### 2.3 Newton / root-finding kernel

Several remaining surfaces (QFM augmented-Lagrangian subproblems,
VMEC `theta_pest -> theta_vmec`, exact-Boozer scaling-limit probes)
need a JAX-native root iteration. We add `simsopt.jax_core._root.py`
with two explicit contracts:

- `newton_scan_fixed_iters(residual, x0, *, max_iter, jac=None)`
  implemented with `jax.lax.scan` so the iteration count is static,
  loop-carried shapes stay fixed, and ordinary autodiff remains
  available when differentiating through the finite iteration itself is
  the intended contract. Default Jacobian via `jax.jacfwd` when none is
  supplied.
- `newton_with_implicit_vjp(residual, x0, params, *, max_iter, tol, jac=None)`
  wrapping the converged solve in `jax.custom_vjp`. This is reverse-mode
  only: `custom_vjp` precludes forward-mode AD. The backward rule solves
  the IFT adjoint
  $(\partial r / \partial x)^T \lambda = \bar{x}$ and returns
  $-(\partial r / \partial p)^T \lambda$ for differentiable parameters.
  Do not use `(I - J)^T` unless the residual has first been rewritten as
  a fixed-point map `x - g(x, p)`.
  `jax.lax.while_loop` may only appear behind this custom-VJP boundary;
  it is not used as an ordinary reverse-mode-differentiated loop.

This kernel is the SSOT for any inner root-find; existing M5 wrappers
already encode the IFT, so this just generalises it.

### 2.4 Solver-adapter pattern

For every new wrapper, follow the same pattern that worked for
`BoozerSurfaceJAX` / `BoozerResidualJAX`:

1. Pure-function kernel in `simsopt.jax_core/<topic>.py` (no
   Optimizable, no host mutation).
2. Public Optimizable adapter in `simsopt.<subpackage>/<name>_jax.py`
   that captures only frozen state in JIT closures.
3. Backend selection inside the adapter is *only* via
   `simsopt.backend.get_backend_mode()` — never via raw env-var reads.
4. Unit tests under `tests/<subpackage>/test_<name>_jax.py`, each
   citing an independent oracle.

### 2.5 Parity-ladder lane assignments

Each new surface declares the parity lane it claims, ahead of the
test writing:

| Surface | Forward-value lane | Derivative lane |
|---|---|---|
| `compute_trapped_fraction_jax` | `direct_kernel` (`rtol=1e-10` only at same quadrature) | `derivative_heavy` (`first_derivative_rtol=1e-8`, `first_derivative_atol=1e-10`) |
| `j_dot_B_Redl_jax` | `direct_kernel` against the upstream CPU implementation, called directly rather than copied into a test helper | `derivative_heavy` |
| `vmec_compute_geometry_jax` | `direct_kernel` against `vmec_compute_geometry` host output | `derivative_heavy` for first derivatives wrt frozen-state DOFs |
| `vmec_fieldlines_jax` | `direct_kernel` (positions) + `branch_stable_resolve` for theta_vmec | `derivative_heavy` for reverse-mode products involving $\nabla\alpha$, $\nabla\psi$ |
| Profile classes | `direct_kernel` (`rtol=1e-12` on same machine) | `derivative_heavy` |
| `QfmSurfaceJAX` penalty path | `ls_wrapper_gradient` (fixed-state objective value+gradient, `rtol=1e-10`) | `fd_gradient` for outer-loop convergence diagnostics |
| `QfmSurfaceJAX` augmented-Lagrangian | `branch_stable_resolve` | `fd_gradient` |
| `least_squares_serial_solve_jax` | reduction matches `least_squares_serial_solve` to `rtol=1e-12` on shared seed | n/a |
| `least_squares_mpi_solve_jax` | reduction matches MPI reference on shared seed | n/a |
| PM/wireframe live loops | `direct_kernel` for inner step value | `derivative_heavy` for inner gradient |

---

## 3. Milestone N1 — Bootstrap (`compute_trapped_fraction`, `j_dot_B_Redl`)

### 3.1 Rationale and purpose

The Redl bootstrap chain is a pure flux-function calculation: it
consumes `(modB, sqrtg)` on a flux-surface grid and a handful of
profiles, and emits `<J·B>` per surface. None of the math requires C++
support; it was simply left in NumPy because the bootstrap workflow
has not been on the critical path. Porting it now unlocks two things:

- a single-stage objective term that includes a Redl current target,
  fully differentiable end-to-end on GPU (so the outer optimizer sees
  one unified JAX gradient tape);
- a clean MHD-evaluation lane that does not bounce out of the JAX
  runtime just to compute a per-surface diagnostic.

The trapped-fraction extrema search (`scipy.optimize.minimize` on a
spline) is the only piece that needs care; the rest is straight
`np → jnp` substitution.

### 3.2 Detailed implementation plan

- [ ] **N1.1 — `simsopt.jax_core/mhd_bootstrap.py`**
  - [ ] Pure-function `compute_trapped_fraction_jax(modB, sqrtg)` that
    accepts 2-D `(ntheta, ns)` and 3-D `(ntheta, nphi, ns)` arrays.
    Implement extrema search by replaying the same one-shot spline
    representation used by the CPU path: a 1-D cubic interpolant for
    `(ntheta, ns)` and a tensor-product 2-D spline for
    `(ntheta, nphi, ns)`. Use a deterministic fixed-iteration bounded
    Newton/local-quadratic search initialized from the grid extrema.
    The acceptance claim is parity of `Bmin`, `Bmax`, and downstream
    `f_t` against the upstream CPU function; do not claim SciPy
    optimizer byte identity.
  - [ ] Replace `scipy.integrate.quad` with a fixed-node quadrature
    rule whose node count is a static argument so the JIT cache key
    remains stable. Validate the quadrature error budget against the
    upstream `quad` output on representative surfaces before assigning
    the `direct_kernel` tolerance.
  - [ ] `jax.vmap` over the `js` (surface) axis. Static-shape
    invariants tracked through `mnmax` / `ntheta` / `nphi` ints.
- [ ] **N1.2 — `simsopt.jax_core/redl_current.py`**
  - [ ] Port `j_dot_B_Redl` line-by-line: `np → jnp`, replace
    `Struct(**locals())` with a `dataclasses.dataclass(frozen=True)`
    pytree (`RedlDetailsJAX`) so the return value is jit-friendly.
    Helicity dispatch (`helicity_n in {0, ±1}`) becomes a static
    argument.
  - [ ] Accept profile *arrays* `(ne, Te, Ti, Zeff)` evaluated on `s`
    rather than `Profile` callables. The public Optimizable wrapper
    in N1.3 is responsible for evaluating the profile classes on the
    `s` grid once.
- [ ] **N1.3 — `simsopt.mhd/bootstrap_jax.py`** (new public module)
  - [ ] `RedlBootstrapJAX(Optimizable)` adapter mirroring the existing
    `RedlGeomVmec` / `RedlGeomBoozer` contract: it depends on the
    profile classes and on a geometry source (either
    `RedlGeomVmecFrozen` or `RedlGeomBoozerFrozen` — defined under
    milestone N3) and exposes `.J()` returning the JAX-evaluated
    `<J·B>` array plus `.dJ_by_d<…>()` accessors.
  - [ ] Keep the CPU `j_dot_B_Redl` path intact for the public/native
    lane; the new `_jax` namespace is selected via
    `simsopt.backend.is_jax_backend()`.
- [ ] **N1.4 — Tests**
  - [ ] `tests/mhd/test_bootstrap_jax.py`:
    - [ ] `compute_trapped_fraction_jax` vs `compute_trapped_fraction`
      at `rtol=1e-10` for the same `(modB, sqrtg)` 2-D and 3-D inputs.
    - [ ] `j_dot_B_Redl_jax` vs `j_dot_B_Redl` for the pinned
      `tests/mhd/test_bootstrap.py` fixture and for the
      `helicity_n ∈ {0, +1, -1}` cases.
    - [ ] Finite-difference vs `jax.grad` of `<J·B>` wrt
      profile coefficients (`derivative_heavy` lane).
- [ ] **N1.5 — Acceptance**
  - [ ] Lint+format+mypy clean on touched files.
  - [ ] All N1.4 tests pass in `JAX_PLATFORM_NAME=cpu`.
  - [ ] Subset run on `JAX_PLATFORM_NAME=cuda` (single GPU, smoke).
  - [ ] `transfer_guard("disallow")` sweep over the inner loop is
    quiet (one host scalar at most per outer call, materialised by
    the public wrapper).

---

## 4. Milestone N2 — Profile classes (`ProfilePolynomial`, `ProfileSpline`, `ProfileScaled`, `ProfilePressure`)

### 4.1 Rationale and purpose

The Redl chain in N1 consumes profile callables. To keep the entire
tape JAX-native we need JAX-friendly profiles whose `.f(s)` and
`.dfds(s)` work under `jax.jit`, `jax.vmap`, and on CUDA. These are
the smallest pure-math wrappers in the audit; the only design choice
is what to do with `ProfileSpline`, which is currently backed by
`scipy.interpolate.InterpolatedUnivariateSpline`.

### 4.2 Detailed implementation plan

- [ ] **N2.1 — `simsopt.jax_core/profiles.py`**
  - [ ] `profile_polynomial_value(coeffs, s)`, `profile_polynomial_dfds(coeffs, s)`
    via an explicit Horner evaluator over reversed coefficients plus
    analytic differentiation of the coefficient vector. Match the
    `numpy.polynomial.polynomial` convention (ascending powers) used
    by `ProfilePolynomial`; do not call `jnp.polyval` directly unless
    the coefficient order is reversed at the same SSOT boundary.
  - [ ] `profile_scaled_value(scale, base_value)` and
    `profile_scaled_dfds(scale, base_dfds)` — pure scalar
    multiplications.
  - [ ] `profile_pressure_value(pairs)` and
    `profile_pressure_dfds(values_pairs, dfds_pairs)` mirroring the
    upstream rule exactly:
    `f(s) = Σ_j f_{2j}(s) f_{2j+1}(s)` and
    `df/ds = Σ_j (df_{2j}/ds * f_{2j+1} + f_{2j} * df_{2j+1}/ds)`.
  - [ ] `profile_spline_value(knots, coeffs, degree, s)` and
    `profile_spline_dfds(...)` using the spline kernel introduced in
    §2.2. Restrict the public API to degrees in `{1, 2, 3, 4, 5}` and
    *fit on the host once* using the existing
    `InterpolatedUnivariateSpline` to source the coefficient arrays;
    the JAX path never re-fits.
- [ ] **N2.2 — `simsopt.mhd/profiles_jax.py`** (new public module)
  - [ ] `ProfilePolynomialJAX(Optimizable)`,
    `ProfileScaledJAX(Optimizable)`,
    `ProfilePressureJAX(Optimizable)`,
    `ProfileSplineJAX(Optimizable)` mirroring the CPU classes
    one-for-one with the same constructor signatures. `local_full_x`
    DOF semantics carry over verbatim.
  - [ ] Each `.f(s)` / `.dfds(s)` accepts and returns `jax.Array` in
    JAX modes. The legacy CPU classes remain the NumPy-returning public
    contract. Any host materialization happens only at the explicit
    Optimizable wrapper boundary via `jax.device_get`, not inside the
    pure profile kernels.
- [ ] **N2.3 — Tests**
  - [ ] `tests/mhd/test_profiles_jax.py`:
    - [ ] Identity-with-CPU for each subclass at the same DOF state
      (`rtol=1e-12` on same machine for polynomial, scaled, pressure;
      `rtol=1e-10` for spline at knots, `rtol=1e-8` off-knot, both
      bounded by SciPy spline truncation error).
    - [ ] `jax.grad` of `.f(s)` wrt DOF coefficients vs analytic
      result and finite difference.
    - [ ] `vmap` over `s` axis and over DOF axis returns
      shape-correct outputs.
- [ ] **N2.4 — Acceptance**
  - [ ] As §1.3 plus: `ProfileSpline` parity at-knot matches CPU
    spline within float64 round-off; off-knot bound documented and
    asserted by tests.

---

## 5. Milestone N3 — `vmec_compute_geometry` (frozen-state JAX path)

### 5.1 Rationale and purpose

`vmec_compute_geometry` is the workhorse geometry routine for
single-stage MHD post-processing. It is presently held together by
~700 lines of `np.zeros`/`InterpolatedUnivariateSpline`/Fourier loops
acting on a live `vmec_splines` struct. Porting it unlocks a JAX
gradient through MHD diagnostics — which is the primary blocker for
making the Redl objective live inside the single-stage tape — and
shows that we can take a VMEC-state-heavy routine off the
`simsoptpp` path entirely after freezing the spline coefficients.

### 5.2 Detailed implementation plan

- [ ] **N3.1 — Spline-coefficient freezing helper**
  - [ ] `simsopt.mhd/_vmec_frozen.py` adds
    `vmec_freeze_splines(vmec_or_splines) -> VmecFrozenSplineState`.
    Calls existing `vmec_splines(vmec)` once on the host, then
    extracts `(t, c, k)` from each `InterpolatedUnivariateSpline` and
    stores `xm`, `xn`, `xm_nyq`, `xn_nyq`, `stellsym`, `mnmax`,
    `mnmax_nyq`, `nfp`, `Aminor_p`, `phiedge`, `pressure`, `iota`,
    and every symmetric / asymmetric VMEC spline family consumed by
    `vmec_compute_geometry` as JAX pytree leaves / static fields.
- [ ] **N3.2 — JAX spline-evaluation kernel**
  - [ ] `simsopt.jax_core/_spline_utils.py` implements
    `bspline_eval` and `bspline_deriv` per §2.2, with `jax.vmap` over
    `mnmax` and the query `s` axis. Bench against
    `InterpolatedUnivariateSpline` at the half- and full-grid points
    for parity (`rtol=1e-12` on same machine).
- [ ] **N3.3 — Pure-function geometry kernel**
  - [ ] `simsopt.jax_core/vmec_geometry.py` defines
    `vmec_compute_geometry_jax(frozen_state, s, theta_vmec, phi, phi_center)`
    returning a frozen `VmecGeometryResultsJAX` pytree.
  - [ ] All Fourier-sum loops over `mnmax` and `mnmax_nyq` rewritten
    as `jnp.einsum` contractions over `(mnmax, ns, ntheta, nphi)`
    using broadcasting; this matches existing patterns in
    `simsopt.jax_core/surface_fourier_kernels.py`.
  - [ ] Metric-tensor and basis-vector computations stay in
    `jnp.einsum` form; no per-surface Python loops.
  - [ ] Output is a single pytree mirroring the field names of the
    existing `VmecGeometryResults`. Reuse field-name declarations
    via a shared `dataclass` so the CPU and JAX dataclasses cannot
    drift in field order.
- [ ] **N3.4 — Public wrapper**
  - [ ] `simsopt.mhd/vmec_diagnostics_jax.py` exports
    `vmec_compute_geometry_jax(vs, s, theta, phi, phi_center=0.0)`
    that accepts either a `Vmec` object (calls `vmec_freeze_splines`
    once), a `vmec_splines` Struct (also freezes), or a
    `VmecFrozenSplineState` directly. Output identical to the CPU
    `vmec_compute_geometry` modulo `jnp.ndarray` types.
- [ ] **N3.5 — Tests**
  - [ ] `tests/mhd/test_vmec_compute_geometry_jax.py`:
    - [ ] Identity-with-CPU on the existing
      `tests/test_files/wout_li383_low_res.nc` fixture for two `s`
      values × `(nphi, ntheta)` = `(8, 16)`. Tolerance: same-machine
      `rtol=1e-10`, `atol=1e-12`.
    - [ ] First-derivative parity via `jax.jacfwd` vs centered
      finite difference (`derivative_heavy` lane).
    - [ ] Stellsym-on and stellsym-off branches each get coverage.
- [ ] **N3.6 — Acceptance**
  - [ ] All N3.5 tests pass on CPU; subset runs on CUDA.
  - [ ] `transfer_guard("disallow")` sweep over the inner loop quiet.
  - [ ] Bench note recorded in
    `docs/vmec_compute_geometry_jax_bench_<date>.md` with HLO node
    count and one-call wall-time.

---

## 6. Milestone N4 — `vmec_fieldlines` (Newton + reuses N3)

### 6.1 Rationale and purpose

The field-line routine wraps N3 with a Newton solve that maps
`theta_pest -> theta_vmec` per surface and per `alpha`; it is not an
ODE integration path. Once N3 is in, this milestone is mostly a Newton
driver plus a thin reorganisation of inputs. The IFT pattern (§2.3)
gives reverse-mode gradients through the inner solve without
back-propagating the iteration itself, matching the already-validated
M5 IFT pattern. Forward-mode through this implicit solve is out of
scope unless a separate `scan`-differentiated finite-iteration contract
is added.

### 6.2 Detailed implementation plan

- [ ] **N4.1 — Newton kernel**
  - [ ] Re-use `simsopt.jax_core/_root.py` (§2.3) with the residual

      $$ r(\theta_v) = \theta_{p,\text{target}}
        - (\theta_v + \sum_{mn} \lambda_{mn}(s)
        \sin(m\theta_v - n\phi)) $$

  - [ ] `jax.vmap` over `(ns, nalpha, nl)` so the explicit root helper
    solves the whole tensor of unknowns with one staged kernel. Use
    `newton_with_implicit_vjp` for the reverse-mode contract; use
    `newton_scan_fixed_iters` only for a separately tested
    finite-iteration contract.
- [ ] **N4.2 — Public wrapper**
  - [ ] `simsopt.mhd/vmec_diagnostics_jax.py` adds
    `vmec_fieldlines_jax(vs, s, alpha, theta1d=None, phi1d=None, phi_center=0.0)`.
    Returns the same dataclass as N3 with the field-line-specific
    attributes `nalpha`, `nl`, `alpha`, `theta1d`, `phi1d` appended.
- [ ] **N4.3 — Tests**
  - [ ] `tests/mhd/test_vmec_fieldlines_jax.py`:
    - [ ] Identity-with-CPU at the same `(s, alpha, theta1d)` /
      `(s, alpha, phi1d)` for both branches; tolerance set by the
      inner-Newton residual (`<=1e-12`) on top of N3’s parity bound.
    - [ ] `branch_stable_resolve` lane: re-solve with two warm-starts
      gives identical theta_vmec up to numerical noise.
    - [ ] `derivative_heavy` lane: `jax.grad` of
      `||grad_psi_dot_grad_psi||_2` wrt frozen state matches
      centred FD on the same fixture.
- [ ] **N4.4 — Acceptance**
  - [ ] N4.3 passes on CPU; smoke on CUDA.
  - [ ] No host pulls in the inner Newton (the only allowed
    materialisation is at the public boundary).

---

## 7. Milestone N5 — QFM Surface orchestration

### 7.1 Rationale and purpose

`QfmResidualJAX` already exists
(`src/simsopt/geo/surfaceobjectives_jax.py:858` plus the
`surface_qfm_*_jax_from_dofs` helpers). The remaining gap is the
*outer solver*: `minimize_qfm_penalty_constraints_LBFGS` (SciPy LBFGS-B
on a penalty form) and `minimize_qfm_exact_constraints_SLSQP` (SciPy
SLSQP with an equality constraint). Both currently mutate `surface.x`
on every callback and rely on `surface.x = x` to refresh internal
state; they cannot be JIT-compiled as-is.

Two production-grade options are viable; this plan picks both and lets
the user choose by mode:

- **Penalty path on-device:** keep the same penalty objective but
  drive it with an in-repo fixed-iteration BFGS loop for forward
  solves. Official JAX docs/source state that
  `jax.scipy.optimize.minimize` is BFGS-only and not differentiable
  through the solve, and live strict-transfer probes showed that it
  stages an internal identity matrix under `transfer_guard("disallow")`;
  the QFM solver therefore cannot use it as the production strict-lane
  core. Derivative-through-solve claims remain reserved for a future
  explicit Optimistix / implicit-adjoint route.
- **Equality-constrained path:** wrap the QFM kernel in an
  augmented-Lagrangian outer loop. JAX has no SLSQP, and optimistix
  does not ship a primal-dual SQP. Augmented Lagrangian is the
  canonical drop-in for equality-constrained problems on JAX. The
  current implementation is a forward-solve route only; convergence
  comparison against host SLSQP and derivative-through-solve support
  remain open acceptance gates.

The host SLSQP path stays as the `native_cpu` reference behavior only.
If a JAX QFM method is selected, missing optional dependencies or
non-convergence fail that selected call; they do not silently dispatch
to host SLSQP.

### 7.2 Detailed implementation plan

- [ ] **N5.1 — `simsopt.jax_core/qfm_solver.py`**
  - [x] Pure functions
    `qfm_penalty_solve_jax(spec, coil_set_spec, label, targetlabel,
    constraint_weight, init_dofs, *, max_iter, tol, optimizer)`
    returning `(final_dofs, info_pytree)` are implemented for
    `optimizer="bfgs"` via an in-repo fixed-iteration BFGS loop whose
    line-search and inverse-Hessian state stay in staged JAX arrays.
    The existing
    `surface_qfm_*_jax_from_dofs` functions now delegate to
    `simsopt.jax_core.qfm_solver` as the SSOT. Optional `"lm"` /
    `"optimistix-bfgs"` routes remain unwired and fail closed instead
    of falling back.
  - [x] `qfm_augmented_lagrangian_solve_jax(spec, coil_set_spec,
    label, targetlabel, init_dofs, *, max_outer, inner_max_iter, tol)`
    implementing the Hestenes–Powell augmented Lagrangian with
    multiplier update rule
    $\lambda_{k+1} = \lambda_k + \rho_k (L(x_k) - \text{target})$,
    $\rho_{k+1} = \min(\rho_{\max}, \beta \rho_k)$. Current inner
    step uses the same transfer-guard-clean in-repo BFGS route as the
    penalty path; optional Optimistix inner solves remain open.
- [x] **N5.2 — `simsopt.geo/qfmsurface_jax.py`**
  - [x] `QfmSurfaceJAX` adapter mirroring `QfmSurface`. Constructor
    accepts the JAX field/surface pair; the immutable coil set spec is
    materialised from explicit JAX coil DOFs at the solve boundary so
    current coil DOFs are not snapshotted at construction.
  - [x] `.minimize_qfm_penalty_jax(tol, maxiter, constraint_weight)`
    drives `qfm_penalty_solve_jax`.
  - [x] `.minimize_qfm_exact_jax(tol, maxiter)` drives
    `qfm_augmented_lagrangian_solve_jax`.
  - [x] `.minimize_qfm(...)` dispatches on `method ∈ {"BFGS", "LM",
    "AL"}` and forwards to the JAX paths only when
    `simsopt.backend.is_jax_backend()` returns true. In `native_cpu`,
    `BFGS` and `AL` route to the existing `QfmSurface` reference
    methods. The unwired JAX-only `LM` route fails closed instead of
    auto-retrying with host SLSQP.
  - [x] Surface DOF write-back at the end of each call uses the same
    `s.x = device_get(final_dofs)` pattern we already use for the
    LS / exact Boozer paths.
- [ ] **N5.3 — Tests**
  - [ ] `tests/geo/test_qfmsurface_jax.py`:
    - [x] Public JAX BFGS penalty solve reduces the fixed-state
      QFM penalty on a low-resolution NCSX fixture.
    - [x] `QfmSurfaceJAX.qfm_penalty_constraints(..., derivatives=1)`
      matches the pure value/gradient helper without mutating
      `surface.x`.
    - [x] The in-repo BFGS solver core runs under
      `jax.transfer_guard("disallow")` on the low-resolution NCSX
      fixture.
    - [x] The augmented-Lagrangian wrapper keeps scalar multiplier /
      penalty-weight updates and the inner BFGS call clean under
      `jax.transfer_guard("disallow")` for a one-outer-step smoke.
    - [x] The augmented-Lagrangian result schema pairs public
      `fun=QFM residual` with the QFM objective gradient rather than
      the augmented-objective gradient.
    - [x] The augmented-Lagrangian diagnostics report
      `augmented_value`, `multiplier`, and `penalty_weight` for the
      final inner objective actually minimized, not the next outer-loop
      state.
    - [x] Strict JAX backend dispatch writes final DOFs only after the
      pure solve and does not enter native SLSQP for `method="AL"`.
    - [x] Native dispatch rejects the unwired JAX-only `method="LM"`
      instead of silently routing it to host SLSQP.
    - [x] `simsopt.geo` lazy-exports `QfmSurfaceJAX`.
    - [ ] Penalty path produces residual ≤ host-SciPy LBFGS-B path on
      the existing `tests/geo/test_qfmsurface.py` fixture
      (`ls_wrapper_gradient` lane).
    - [ ] AL path satisfies equality residual $|L - L_{\text{target}}| \le 10^{-6}$
      on the same fixture and matches host SLSQP solution within
      `rtol=1e-6` on the surface DOFs (`branch_stable_resolve`).
    - [ ] `derivative_heavy` lane: fixed-state penalty objective
      gradients match FD. For converged-solve sensitivities, test only
      the Optimistix / implicit-adjoint route; do not claim
      differentiation through the current forward BFGS solve.
- [ ] **N5.4 — Acceptance**
  - [x] Focused CPU evidence: `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
    pytest -q -p no:cacheprovider tests/geo/test_qfmsurface_jax.py`
    passed on 2026-05-19 (`10 passed`).
  - [x] Existing QFM penalty kernel evidence stayed green:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q -p
    no:cacheprovider tests/geo/test_surface_objectives_jax.py::TestQfmPenaltyJAX
    tests/geo/test_qfm.py::QfmSurfaceTests::test_qfm_penalty_constraints_gradient`
    passed on 2026-05-19 (`5 passed, 8 subtests passed`).
  - [x] Local quality evidence: `ruff check --no-cache`,
    `ruff format --check --no-cache`, isolated
    `mypy --cache-dir=/dev/null --follow-imports=skip --ignore-missing-imports`,
    and `git diff --check` passed on the N5 touched files on 2026-05-19.
  - [x] No `surface.x = x` mutations inside the inner solve for the
    implemented BFGS/AL adapter seam; tests assert the surface is
    unchanged while the pure solver is running and only receives final
    device DOFs after the solver returns.
  - [x] Strict transfer-guard evidence: the implemented BFGS solver
    and AL wrapper paths are covered by
    `tests/geo/test_qfmsurface_jax.py::test_qfm_penalty_solve_jax_transfer_guard_clean`
    and
    `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_solve_jax_transfer_guard_clean`.
  - [ ] Full host-SciPy LBFGS-B / host SLSQP convergence comparison
    and CUDA smoke remain open.

---

## 8. Milestone N6 — `solve/serial.py` and `solve/mpi.py` (JAX-aware solvers)

### 8.1 Rationale and purpose

`least_squares_serial_solve` and `least_squares_mpi_solve` are generic
SciPy drivers over arbitrary mutable `Optimizable` graphs. They are
*not* candidates for a wholesale port; they remain the host
orchestrators for the wider simsopt universe. What we *can* do, and
what the audit asks for, is to provide a **JAX-aware least-squares
lane** with a strict traceable-problem protocol that:

- accepts a `TraceableLeastSquaresProblem` / adapter whose residual
  function is pure JAX and whose state is an explicit pytree. Do not
  infer traceability from arbitrary `LeastSquaresProblem` instances;
- uses `optimistix.LevenbergMarquardt` / `optimistix.GaussNewton` on
  the device for the inner step in the optional
  `simsopt[JAX_OPTIMISTIX]` lane;
- replaces the per-DOF finite-difference Jacobian with `jax.jacfwd`
  in the gradient-on lane and with a `shard_map`-parallel forward
  finite-difference pass in the no-gradient lane on a single host
  with multiple GPUs;
- keeps the MPI worker partition for multi-host scenarios. The MPI
  outer loop still owns DOF distribution, but the *inner* gradient
  evaluation can now reside on each rank’s local GPU via JAX.

### 8.2 Detailed implementation plan

- [ ] **N6.1 — `simsopt.solve/serial_jax.py`**
  - [ ] `least_squares_serial_solve_jax(prob, *, optimizer="lm", abs_step=1e-7, rel_step=0.0, **kwargs)`
    mirroring the public signature of `least_squares_serial_solve`.
    Require the traceable-problem adapter explicitly; do not wrap
    arbitrary host `Optimizable` graphs or copy the host driver's
    `try`/large-residual recovery behavior.
  - [ ] Iteration log written to the same `simsopt_<datestr>.dat`
    file format as the host driver so downstream tooling is
    unaffected. The log writer reads JAX scalars only at the boundary
    (one `device_get` per iteration).
  - [ ] `serial_solve_jax` and `constrained_serial_solve_jax` follow
    the same template but call the augmented-Lagrangian wrapper from
    N5 for the constrained case.
- [ ] **N6.2 — Finite-difference Jacobian via `shard_map`**
  - [ ] `simsopt.jax_core/_finite_difference.py` adds
    `forward_jacobian_shard_map(fn, x0, abs_step, rel_step,
    diff_method, mesh)` that distributes the per-DOF perturbations
    across a `shard_map` over an `('dof',)` mesh axis. `mesh` is
    explicit. The single-device sibling is
    `forward_jacobian_vmap(...)`; callers choose one route by backend
    policy rather than relying on implicit route selection.
  - [ ] When `JAX_PLATFORM_NAME=cuda` and `XLA_FLAGS` enables
    real GPU execution, devices are read from `jax.devices()`. CPU
    fake-device tests may use `--xla_force_host_platform_device_count`
    before JAX import, but that is test-only. Static-shape constraint:
    pad the DOF axis to the mesh size inside the explicit sharded
    function and discard padded columns at assembly.
- [ ] **N6.3 — `simsopt.solve/mpi_jax.py`**
  - [ ] `least_squares_mpi_solve_jax(prob, mpi, *, …)` reuses the
    existing `MpiPartition` worker pool but offloads each rank’s
    Jacobian column block to the JAX path of N6.2 — i.e. each rank
    runs `forward_jacobian_shard_map` over its assigned columns
    locally, then `Allgather`s the assembled `J` to rank 0.
  - [ ] `mpi.comm_groups.bcast` of `x` stays exactly as in the host
    driver.
- [ ] **N6.4 — Tests**
  - [ ] `tests/solve/test_serial_jax.py`:
    - [ ] Identity-with-host on the existing
      `tests/objectives/test_least_squares.py` toy problem with
      a `TraceableLeastSquaresProblem` adapter over the JAX-traced
      residual kernel.
    - [ ] `derivative_heavy` lane: `jax.jacfwd` Jacobian matches
      `forward_jacobian_shard_map` output within `rtol=1e-10`.
  - [ ] `tests/solve/test_mpi_jax.py` (gated behind `mpi4py` import):
    - [ ] 2-rank smoke run reproducing the host MPI solver to
      `rtol=1e-12` on the same seed.
- [ ] **N6.5 — Acceptance**
  - [ ] CPU pass on N6.4; CUDA smoke at 1 device.
  - [ ] Multi-GPU smoke for N6.2 is **hardware-gated** (cross-link
    against N30 in `docs/jax_native_round3_curated_todos_2026-05-18.md`).

---

## 9. Milestone N7 — Live PM / wireframe workflows

### 9.1 Rationale and purpose

`simsopt.solve/permanent_magnet_optimization_jax.py`,
`simsopt.solve/wireframe_optimization_jax.py`, and the
`simsopt.jax_core.pm_optimization` / `wireframe` kernels already carry
substantial `jax.lax.scan` coverage for the inner maths. What is not
yet fully JAX-native is the live workflow boundary: output writers
(FAMUS files, VTK, JSON, plots), restart snapshots, pruning/final
adjustment orchestration, and any host callback cadence. The audit
explicitly does not call for byte-identity output writers; it asks that
the **decision logic that changes the numerical state** runs in a
single explicit JAX loop, with host I/O only at documented boundaries.

### 9.2 Detailed implementation plan

- [ ] **N7.1 — `simsopt.jax_core/pm_workflow.py`**
  - [ ] `pm_gpmo_live_loop_jax(state, *, max_steps, prune_rule, stop_rule)`
    wrapping the per-step `GPMO_*` kernels in a fixed-length
    `jax.lax.scan` with an active/done carry. The state
    pytree carries the moment matrix, the active-loop mask, and the
    history slice needed for restart.
  - [ ] Pruning rules expressed as pure functions
    `(state) -> (state, prune_mask)` so the loop body stays
    JIT-friendly.
- [ ] **N7.2 — `simsopt.jax_core/wireframe_workflow.py`**
  - [ ] `gsco_live_loop_jax(state, *, max_steps, params, stop_rule)`
    analogously wraps the existing GSCO inner step.
  - [ ] Final-adjustment phase (currently a host loop) implemented as
    a fixed-length `lax.scan` with an active/done carry derived from
    the residual delta, matching the PM loop contract.
- [ ] **N7.3 — Host-side I/O boundary**
  - [ ] Output writers (FAMUS / VTK / JSON / matplotlib) remain in
    `simsopt.solve/*_optimization_jax.py`. They consume the final
    state pytree via a single `device_get` at the end of the
    workflow. No mid-loop `device_get` calls.
  - [ ] A `record_every: Optional[int]` parameter optionally pulls
    snapshots out of the device tape at fixed cadence for
    downstream plotting; this materialisation cost is paid only
    when the user opts in.
- [ ] **N7.4 — Tests**
  - [ ] `tests/solve/test_pm_workflow_jax.py`:
    - [ ] Live-loop output equals step-by-step host loop output for
      the existing PM fixture at `rtol=1e-12` on identical seeds.
    - [ ] Restart from snapshot reproduces continuation step exactly.
  - [ ] `tests/solve/test_wireframe_workflow_jax.py`:
    - [ ] GSCO live loop reproduces the host loop on the existing
      fixture for `max_steps ∈ {5, 50}`.
- [ ] **N7.5 — Acceptance**
  - [ ] N7.4 passes on CPU; CUDA smoke.
  - [ ] No mid-loop `device_get` calls in the new live loops
    (verified via `jax.transfer_guard("disallow")`).

---

## 10. Cross-Milestone Validation, CI, and Documentation

- [ ] **V1 — Parity-ladder updates.** Each milestone cites an existing
  lane key from `benchmarks/validation_ladder_contract.py` in tests and
  closeout docs. Do not add or rename tolerance lanes unless a surface
  genuinely needs a new tolerance contract; if that happens, update the
  SSOT first and then update the table in §2.5.
- [ ] **V2 — Strict gate sweep.** After every milestone merges, run
  `benchmarks/single_stage_init_parity.py` under `jax_cpu_parity` and
  `jax_gpu_parity` modes to confirm the byte-identity gate is intact.
- [ ] **V3 — Mode-matrix smoke.** A new
  `tests/integration/test_remaining_jax_surfaces_mode_matrix.py`
  imports each new public entrypoint under all six
  `SIMSOPT_BACKEND_MODE` values and a `JAX_PLATFORM_NAME=cuda` smoke.
- [ ] **V4 — CI hooks.** Update `.github/workflows/jax_smoke.yml` so
  the new pure-JAX modules and public wrappers that are documented as
  `simsoptpp`-free are imported under the smoke job. Add a CUDA-gated
  job under whatever runner we use for `jax_gpu_parity` smoke today.
- [ ] **V5 — Docs.**
  - [ ] Extend `docs/source/jax_acceptance.rst` with the bootstrap /
    profiles / VMEC-diagnostics acceptance criteria.
  - [ ] Add a “Frozen VMEC state” section to
    `docs/source/jax_gpu_setup.rst` covering N3.1.
  - [ ] Cross-link this plan from `docs/jax_native_round3_curated_todos_2026-05-18.md`
    once the first milestone (N1) lands.

---

## 11. Risk Register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Frozen VMEC state diverges from live `Vmec.wout` if user mutates the run | medium | high | document frozen state as an immutable snapshot; users must call `vmec_freeze_splines` again after mutation |
| Augmented Lagrangian inner loop fails to converge on stiff equality constraint | medium | medium | expose `rho_max` and `beta` knobs and return a failed solver status for the selected JAX method |
| `shard_map` per-device Jacobian column block requires padding when `dof_size % n_devices != 0` | high | low | pad with no-op columns and discard at the rank-0 assembly step |
| `optimistix.LevenbergMarquardt` does not converge for ill-conditioned QFM problems | low | medium | allow the caller to select `optimistix.BFGS` or the JAX BFGS penalty path explicitly; do not auto-switch solvers |
| CUDA toolchain mismatch for new kernels on Runpod | medium | medium | already tracked under `project_runpod_cuda_block`; reuse the launcher patches landed in `scripts/runpod_single_stage_continuation.py` |
| ProfileSpline off-knot drift between SciPy fit and JAX evaluation | low | medium | reuse SciPy fit coefficients; assert at-knot `rtol=1e-12` and bound off-knot via SciPy's own truncation budget |
| MPI workers idle while JAX kernels saturate a single GPU | medium | low | document the recommended one-rank-per-GPU layout in `docs/source/jax_gpu_setup.rst`; do not change the MPI worker contract |

---

## 12. Sequencing and Dependencies

```
N2 (profiles)   ──┐
                  ├──> N1 (bootstrap)  ──┐
N3.1+N3.2 (frozen state + bspline) ─────┤
                  └──> N3 (vmec_compute_geometry) ──> N4 (vmec_fieldlines)

QfmResidualJAX (already in) ──────────> N5 (QfmSurfaceJAX)

Existing JAX wrappers + sharding ─────> N6 (serial/mpi JAX-aware)

Existing PM/wireframe JAX kernels ────> N7 (live loops)
```

Recommended landing order: **N2 → N1 → N3 → N4 → N5 → N6 → N7.**
N2 and N3.1/N3.2 are the only blocking prerequisites; N5, N6, and N7
are independent and can be parallelised across reviewers.

---

## 13. Success Criteria (overall)

This plan succeeds when:

1. Every row in the §0 audit table has a green checkbox in §3–§9.
2. `simsopt.jax_core` remains `simsoptpp`-free.
3. The byte-identity gate
   (`benchmarks/single_stage_init_parity.py::_pre_newton_census_gate_failures`)
   is intact in both `jax_cpu_parity` and `jax_gpu_parity` modes.
4. CI smoke (`jax_smoke.yml`) imports the new modules cleanly.
5. The `tests/REVIEWER_ORACLE_LINT.md` rule is satisfied for every
   new `test_*_jax_*.py` file shipped by this plan.

---

## 14. Open Questions

- [ ] Do we want to expose a `compute_trapped_fraction_jax` that
  accepts the *frozen* VMEC state directly (skipping
  `(modB, sqrtg)` reconstruction), or keep the current public
  signature only? Recommendation: keep the public signature,
  introduce a private `_jax_core` entrypoint that consumes the
  frozen state.
- [ ] For `ProfileSplineJAX`, do we ship the SciPy-fit coefficients
  alongside the DOF vector (preferred, matches SciPy exactly) or do
  we re-fit on-device with a JAX-native cubic-spline fit (slower,
  introduces a new failure mode)? Recommendation: SciPy fit, store
  coefficients in the pytree leaf.
- [ ] Should the augmented-Lagrangian QFM path expose a *constraint
  Jacobian* hook so users can plug in cheaper closed-form label
  gradients? Recommendation: yes, default to `jax.jacfwd` and accept
  an optional override.
- [ ] Multi-host `shard_map` (across nodes via `jax.distributed`) is
  out of scope for N6 — confirm this is acceptable for the current
  shipping target, or escalate to a separate milestone.

---

*End of plan.*
