# Remaining JAX-Port Surfaces — Implementation Plan (2026-05-19)

- **Branch:** `gpu-purity-stage2-20260405`
- **HEAD reviewed:** `bda7623d7` plus dirty working-tree content on
  2026-05-19. The checked implementation/test evidence below is local
  working-tree evidence, not commit-accurate release proof, until the referenced
  source and test files are tracked at the proof SHA.
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
  - `optimistix` (optimizer dependencies from the `simsopt[JAX]` /
    `simsopt[JAX_GPU]` extras: LM, IndirectLM, Dogleg, BFGS) for
    least-squares / minimization;
    augmented-Lagrangian wrapper for equality-constrained problems.
  - SciPy spline APIs (`InterpolatedUnivariateSpline`,
    `RectBivariateSpline`) for extracting and replaying FITPACK spline
    coefficients through `get_knots()` / `get_coeffs()`.

**Commit-accuracy gate before release/GPU proof:** for every checked evidence
item that references new source or test paths, `git ls-files --error-unmatch
<path>` and `git cat-file -e <proof_sha>:<path>` must succeed. If either fails,
the item remains local working-tree validation only and must not be cited as a
clean committed proof artifact.

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
| 8 | Live PM / wireframe workflows | fixed-state kernels, restartable live loops, and PM/wireframe numerical workflow orchestration are JAX-native; FAMUS/plot/VTK writers remain host-side |

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
| `QfmSurfaceJAX` augmented-Lagrangian | absolute natural-equality KKT success plus objective/label/KKT branch-invariant acceptance; host SLSQP remains a relative diagnostic, not a DOF oracle | `fd_gradient` |
| `least_squares_serial_solve_jax` | reduction matches `least_squares_serial_solve` to `rtol=1e-12` on shared seed | n/a |
| `serial_solve_jax` | scalar-objective solve reaches the host `serial_solve` / analytic shared-seed optimum on an explicit traceable problem; host graph wrapping is rejected | n/a |
| `constrained_serial_solve_jax` | augmented-Lagrangian equality solve reaches host SLSQP / analytic shared-seed optimum on an explicit traceable problem; host SLSQP fallback and host graph wrapping are rejected | n/a |
| `forward_jacobian_shard_map` / `traceable_least_squares_mpi_jacobian` | `derivative_heavy` against `jax.jacfwd` and vectorized finite-difference oracles; real CUDA multi-device proof uses explicit `jax.devices("cuda")` mesh | n/a |
| PM fixed-state JAX solve wrappers | `direct_kernel` for closed-form helpers; `pm_mwpgp_fixed_step` for MWPGP state traces | n/a |
| PM/wireframe live-loop JAX workflows | `reporting_contract` for host-boundary state/log/restart invariants; fixed-step math is covered by `direct_kernel` wrapper tests | n/a |
| `least_squares_mpi_solve_jax` | traceable-JAX MPI solve reaches the analytic shared-seed optimum; host MPI reference parity remains open | n/a |
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

- [x] **N1.1 — `simsopt.jax_core/mhd_bootstrap.py`**
  - [x] Pure-function `compute_trapped_fraction_jax(modB, sqrtg)` that
    accepts 2-D `(ntheta, ns)` and 3-D `(ntheta, nphi, ns)` arrays.
    Implement extrema search by replaying the same one-shot spline
    representation used by the CPU path: a 1-D cubic interpolant for
    `(ntheta, ns)` and a tensor-product 2-D spline for
    `(ntheta, nphi, ns)`. Use a deterministic fixed-iteration bounded
    Newton/local-quadratic search initialized from the grid extrema.
    The acceptance claim is parity of `Bmin`, `Bmax`, and downstream
    `f_t` against the upstream CPU function; do not claim SciPy
    optimizer byte identity. Evidence:
    `tests/mhd/test_bootstrap_jax.py::test_compute_trapped_fraction_jax_matches_cpu_2d_and_3d`.
  - [x] Replace `scipy.integrate.quad` with a fixed-node quadrature
    rule whose node count is a static argument so the JIT cache key
    remains stable. Validate the quadrature error budget against the
    upstream `quad` output on representative surfaces before assigning
    the `direct_kernel` tolerance. Current CPU evidence records the
    fixed-quadrature vs. adaptive-`quad` `f_t` bound at `rtol=1e-8`,
    `atol=5e-11`; tighter `direct_kernel` tolerance is only claimed
    for extrema and flux-surface averages.
  - [x] `jax.vmap` over the `js` (surface) axis. Static-shape
    invariants tracked through `mnmax` / `ntheta` / `nphi` ints.
- [x] **N1.2 — `simsopt.jax_core/redl_current.py`**
  - [x] Port `j_dot_B_Redl` line-by-line: `np → jnp`, replace
    `Struct(**locals())` with a `dataclasses.dataclass(frozen=True)`
    pytree (`RedlDetailsJAX`) so the return value is jit-friendly.
    Helicity dispatch (`helicity_n in {0, ±1}`) becomes a static
    argument; traced helicity values are rejected by the kernel's
    Python `int(helicity_n)` boundary.
  - [x] Accept profile *arrays*
    `(ne, Te, Ti, Zeff, d_ne_d_s, d_Te_d_s, d_Ti_d_s)` evaluated on
    `s` rather than `Profile` callables. The public Optimizable wrapper
    in N1.3 is responsible for evaluating the profile classes on the
    `s` grid once. Evidence:
    `tests/mhd/test_bootstrap_jax.py::test_j_dot_B_Redl_jax_matches_cpu_details_for_helicity_cases`.
- [x] **N1.3 — `simsopt.mhd/bootstrap_jax.py`** (new public module)
  - [x] Functional public wrapper `j_dot_B_Redl_jax(...)` evaluates
    profile objects once at the host boundary, then calls the pure
    array kernel. It also exports `j_dot_B_Redl_jax_from_arrays` for
    callers that already own frozen profile arrays.
  - [x] `RedlBootstrapJAX(Optimizable)` adapter depends on the profile
    classes and a Redl geometry source implementing the existing
    `RedlGeomVmec` / `RedlGeomBoozer` data contract. It exposes `.J()`
    returning the JAX-evaluated `<J·B>` array, `.details()`, and
    explicit profile-DOF Jacobian accessors
    `.dJ_by_dne_dofs()`, `.dJ_by_dTe_dofs()`,
    `.dJ_by_dTi_dofs()`, and `.dJ_by_dZeff_dofs()`.
    Evidence:
    `tests/mhd/test_bootstrap_jax.py::test_redl_bootstrap_jax_adapter_matches_public_wrapper_and_mhd_export`
    and
    `tests/mhd/test_bootstrap_jax.py::test_redl_bootstrap_jax_density_dof_jacobian_matches_finite_difference`.
  - [x] Keep the CPU `j_dot_B_Redl` path intact for the public/native
    lane. The `_jax` namespace is explicit for this Redl-only slice;
    automatic `simsopt.backend.is_jax_backend()` dispatch remains out
    of scope for N1.3 and is tracked with later end-to-end backend
    routing work.
- [x] **N1.4 — Tests**
  - [x] `tests/mhd/test_bootstrap_jax.py`:
    - [x] `compute_trapped_fraction_jax` vs `compute_trapped_fraction`
      for the same `(modB, sqrtg)` 2-D and 3-D inputs. Extrema/FSA use
      tight direct-kernel tolerances; `f_t` uses the measured
      fixed-quadrature vs adaptive-`quad` budget above. Includes
      non-analytic off-grid 2-D and 3-D fixtures to guard against
      grid-extrema false positives.
    - [x] `j_dot_B_Redl_jax` vs `j_dot_B_Redl` for the pinned
      `tests/mhd/test_bootstrap.py` fixture and for the
      `helicity_n ∈ {0, +1, -1}` cases.
    - [x] Finite-difference vs `jax.grad` of `<J·B>` wrt
      profile coefficients (`derivative_heavy` lane).
    - [x] `RedlBootstrapJAX` adapter `.J()` and `.details()` match the
      functional public wrapper, and the density profile-DOF Jacobian
      matches centered finite difference.
    - [x] Accepted CPU `ProfilePolynomial` and numeric/default
      `Zeff` inputs remain live profile dependencies for `.J()`, while
      synchronized JAX polynomial helpers make profile-DOF accessors
      valid.
    - [x] Accepted CPU profile dependencies are not snapshotted:
      mutating the original `ProfilePolynomial` after adapter
      construction is reflected by `.J()`.
- [x] **N1.5 — Acceptance**
  - [x] Lint+format+mypy clean on touched files.
  - [x] All implemented N1.4 tests pass in `JAX_PLATFORM_NAME=cpu`.
  - [x] Subset run on `JAX_PLATFORM_NAME=cuda` (single GPU, smoke).
    Perlmutter debug job `53204536` (`ljax-mhd`) requested
    `--gpus-per-node=1` under `-q debug -C gpu`, ran with
    `JAX_PLATFORMS=cuda,cpu` and
    `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`, and passed
    `tests/mhd/test_bootstrap_jax.py`, `tests/mhd/test_profiles_jax.py`,
    `tests/mhd/test_vmec_compute_geometry_jax.py`,
    `tests/mhd/test_vmec_fieldlines_jax.py`, and
    `tests/mhd/test_vmec_frozen.py` on CUDA on 2026-05-20
    (`60 passed in 181.65s`).
  - [x] `transfer_guard("disallow")` sweep over the inner loop is
    quiet (one host scalar at most per outer call, materialised by
    the public wrapper).
  - [x] CPU evidence: `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
    pytest -q -p no:cacheprovider
    tests/mhd/test_bootstrap.py::BootstrapTests::test_compute_trapped_fraction
    tests/mhd/test_bootstrap.py::BootstrapTests::test_Redl_second_pass
    tests/mhd/test_bootstrap_jax.py` passed on 2026-05-19 (`17 passed`).
  - [x] Local quality evidence: `ruff check`,
    `ruff format --check`, and isolated
    `mypy --cache-dir=/dev/null --follow-imports=skip --ignore-missing-imports`
    passed on
    `src/simsopt/jax_core/mhd_bootstrap.py`,
    `src/simsopt/jax_core/redl_current.py`,
    `src/simsopt/mhd/bootstrap_jax.py`, and
    `tests/mhd/test_bootstrap_jax.py` on 2026-05-19.
  - [x] `jax.transfer_guard("disallow")` evidence: the jitted pure
    trapped-fraction and Redl array kernels passed transfer-guard tests
    in `tests/mhd/test_bootstrap_jax.py`.

---

## 4. Milestone N2 — Profile classes (`ProfilePolynomial`, `ProfileSpline`, `ProfileScaled`, `ProfilePressure`)

### 4.1 Rationale and purpose

The Redl chain in N1 consumes profile callables. To keep the entire
tape JAX-native we need JAX-friendly profile kernels and explicit-state
profile methods that work under `jax.jit`, `jax.vmap`, and on CUDA.
The mutable `Optimizable` wrappers remain public host-boundary objects:
their `.f(s)` / `.dfds(s)` methods read the current DOF state at call
time, while reusable JIT callers pass DOFs or spline coefficient state
explicitly through `*_from_dofs` / `*_from_state` methods. These are the
smallest pure-math wrappers in the audit; the only design choice is
what to do with `ProfileSpline`, which is currently backed by
`scipy.interpolate.InterpolatedUnivariateSpline`.

### 4.2 Detailed implementation plan

- [x] **N2.1 — `simsopt.jax_core/profiles.py`**
  - [x] `profile_polynomial_value(coeffs, s)`, `profile_polynomial_dfds(coeffs, s)`
    via an explicit Horner evaluator over reversed coefficients plus
    analytic differentiation of the coefficient vector. Match the
    `numpy.polynomial.polynomial` convention (ascending powers) used
    by `ProfilePolynomial`; do not call `jnp.polyval` directly unless
    the coefficient order is reversed at the same SSOT boundary.
  - [x] `profile_scaled_value(scale, base_value)` and
    `profile_scaled_dfds(scale, base_dfds)` — pure scalar
    multiplications.
  - [x] `profile_pressure_value(pairs)` and
    `profile_pressure_dfds(values_pairs, dfds_pairs)` mirroring the
    upstream rule exactly:
    `f(s) = Σ_j f_{2j}(s) f_{2j+1}(s)` and
    `df/ds = Σ_j (df_{2j}/ds * f_{2j+1} + f_{2j} * df_{2j+1}/ds)`.
  - [x] `profile_spline_value(knots, coeffs, degree, s)` and
    `profile_spline_dfds(...)` using the spline kernel introduced in
    §2.2. Restrict the public API to degrees in `{1, 2, 3, 4, 5}` and
    *fit on the host once* using the existing
    SciPy FITPACK path to source the coefficient arrays; the JAX
    value/derivative path never re-fits. Evidence:
    `tests/mhd/test_profiles_jax.py::test_profile_spline_jax_fits_once_per_dof_state`.
- [x] **N2.2 — `simsopt.mhd/profiles_jax.py`** (new public module)
  - [x] `ProfilePolynomialJAX(Optimizable)`,
    `ProfileScaledJAX(Optimizable)`,
    `ProfilePressureJAX(Optimizable)`,
    `ProfileSplineJAX(Optimizable)` mirroring the CPU classes
    one-for-one with the same constructor signatures. `local_full_x`
    DOF semantics carry over verbatim.
  - [x] Each `.f(s)` / `.dfds(s)` accepts and returns `jax.Array` in
    JAX modes. The legacy CPU classes remain the NumPy-returning public
    contract. Reusable JIT callers use explicit-state methods
    (`f_from_dofs`, `dfds_from_dofs`, `f_from_state`, `dfds_from_state`,
    `f_from_values`, `dfds_from_values`) so DOF updates are normal JAX
    arguments rather than closed-over Python object state.
- [x] **N2.3 — Tests**
  - [x] `tests/mhd/test_profiles_jax.py`:
    - [x] Identity with independent oracles at the same DOF state:
      closed-form formulas for polynomial/scaled/pressure and SciPy
      FITPACK `splrep`/`splev` for spline replay (`rtol=1e-12` for
      polynomial/scaled/pressure; `rtol=1e-10` for spline tck replay;
      `rtol=1e-8` for wrapper off-knot spline replay).
    - [x] `jax.grad` of pure profile kernels wrt explicit coefficient
      arrays vs analytic result and finite difference. Direct
      differentiation through mutable `Optimizable.f(s)` state is not
      claimed; wrappers project the current host DOF state into pure
      kernels at the public boundary.
    - [x] `vmap` over `s` axis and over DOF axis returns
      shape-correct outputs; explicit-state JIT calls track DOF updates
      by passing the updated DOFs / FITPACK state as arguments.
    - [x] Direct `jax.jit(profile_spline_value)` and
      `jax.jit(profile_spline_dfds)` replay SciPy FITPACK tck state
      without requiring callers to mark `degree` static; invalid
      degrees fail instead of silently taking the `lax.switch` clamp.
- [x] **N2.4 — Acceptance**
  - [x] As §1.3 plus: `ProfileSplineJAX` parity at-knot matches the
    SciPy FITPACK tck replay within float64 round-off; off-knot bound
    documented and asserted by
    `tests/mhd/test_profiles_jax.py::test_profile_spline_jax_wrapper_tracks_fitpack_after_dof_update`.
  - [x] CPU evidence: `pytest -q tests/mhd/test_profiles.py tests/mhd/test_profiles_jax.py`
    passed on 2026-05-19 (`25 passed`).
  - [x] Local quality evidence: `ruff check`, `ruff format --check`,
    and isolated `mypy --follow-imports=skip --ignore-missing-imports`
    passed on the N2 touched files on 2026-05-19.
  - [x] `jax.transfer_guard("disallow")` profile-kernel sweep passed on
    CPU for the JIT-compiled polynomial/FITPACK-spline kernels and the
    explicit-state JIT update path on 2026-05-19.
  - [x] `simsopt.mhd.profiles_jax` imports without `simsoptpp` present;
    the package-level MHD imports now use lazy exports so the JAX profile
    submodule is not blocked by VMEC/geometry imports.
  - [x] CUDA smoke, full CLAUDE.md regression suite, and
    `private_optimizer_runtime` lane are complete as a combined gate.
    CUDA smoke is proven by Perlmutter debug job `53204536`
    (`60 passed in 181.65s`). Local public CLAUDE.md regression
    evidence passed on 2026-05-20:
    `tests/test_jax_import_smoke.py`,
    `tests/field/test_biotsavart_jax.py`,
    `tests/geo/test_surface_fourier_jax.py`,
    `tests/geo/test_boozer_residual_jax.py`,
    `tests/objectives/test_integral_bdotn_jax.py`,
    `tests/geo/test_boozer_derivatives_jax.py`,
    `tests/geo/test_boozersurface_jax.py`, and
    `tests/integration/test_jax_native_path.py` passed under the
    public lane marker filter (`893 passed, 119 skipped in 1180.12s`).
    The private optimizer runtime packet passed under the private lane
    marker filter (`54 passed, 227 deselected in 745.28s`). Benchmark
    helper regressions also passed (`272 passed, 2 skipped in 3.56s`).

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

- [x] **N3.1 — Spline-coefficient freezing helper**
  - [x] `simsopt.mhd/_vmec_frozen.py` adds
    `vmec_freeze_splines(vmec_or_splines) -> VmecFrozenSplineState`.
    Calls existing `vmec_splines(vmec)` once on the host, then
    extracts `(t, c, k)` from each `InterpolatedUnivariateSpline` and
    stores `xm`, `xn`, `xm_nyq`, `xn_nyq`, `stellsym`, `mnmax`,
    `mnmax_nyq`, `nfp`, `Aminor_p`, `phiedge`, `pressure`, `iota`,
    and every symmetric / asymmetric VMEC spline family consumed by
    `vmec_compute_geometry` as JAX pytree leaves / static fields.
    Evidence:
    `tests/mhd/test_vmec_frozen.py::test_vmec_freeze_splines_metadata_and_pytree_contract`.
- [x] **N3.2 — JAX spline-evaluation kernel**
  - [x] `simsopt.jax_core/_spline_utils.py` implements
    `bspline_eval_1d` and `bspline_deriv_1d` per §2.2. The VMEC
    frozen-state helpers `vmec_spline_eval` and
    `vmec_spline_deriv_eval` apply them over scalar splines and
    mode-table splines with `jax.vmap`. Bench against
    `InterpolatedUnivariateSpline` at the half- and full-grid points
    for parity (`rtol=1e-12` on same machine). Evidence:
    `tests/mhd/test_vmec_frozen.py::test_vmec_frozen_spline_eval_matches_cpu_splines`
    and
    `tests/mhd/test_vmec_frozen.py::test_vmec_frozen_spline_derivative_matches_cpu_derivative_splines`.
- [x] **N3.3 — Pure-function geometry kernel**
  - [x] `simsopt.jax_core/vmec_geometry.py` defines
    `vmec_compute_geometry_jax(frozen_state, s, theta_vmec, phi, phi_center)`
    returning a frozen `VmecGeometryResultsJAX` pytree.
  - [x] All Fourier-sum loops over `mnmax` and `mnmax_nyq` rewritten
    as `jnp.einsum` contractions over `(mnmax, ns, ntheta, nphi)`
    using broadcasting; this matches existing patterns in
    `simsopt.jax_core/surface_fourier_kernels.py`.
  - [x] Metric-tensor and basis-vector computations stay in
    `jnp.einsum` form; no per-surface Python loops.
  - [x] Output is a single pytree mirroring the field names and field
    order of the existing `VmecGeometryResults`; field-order drift is
    covered by
    `tests/mhd/test_vmec_compute_geometry_jax.py::test_vmec_geometry_result_jax_fields_match_cpu_dataclass`.
- [x] **N3.4 — Public wrapper**
  - [x] `simsopt.mhd/vmec_diagnostics_jax.py` exports
    `vmec_compute_geometry_jax(vs, s, theta, phi, phi_center=0.0)`
    that accepts either a `Vmec` object (calls `vmec_freeze_splines`
    once), a `vmec_splines` Struct (also freezes), or a
    `VmecFrozenSplineState` directly. Output identical to the CPU
    `vmec_compute_geometry` modulo `jnp.ndarray` types.
    Evidence:
    `tests/mhd/test_vmec_compute_geometry_jax.py::test_public_vmec_compute_geometry_jax_accepts_vmec_splines_and_frozen_state`
    and
    `tests/mhd/test_vmec_compute_geometry_jax.py::test_public_vmec_compute_geometry_jax_lazy_mhd_export`.
- [x] **N3.5 — Tests**
  - [x] `tests/mhd/test_vmec_frozen.py`:
    - [x] Frozen metadata and pytree leaves match the upstream
      `vmec_splines` Struct.
    - [x] Frozen scalar and mode-table FITPACK state replays CPU
      `InterpolatedUnivariateSpline` values at `rtol=1e-12`.
    - [x] Stellsym asymmetric tables are zero-valued while retaining
      the same mode-table shape needed by later JAX kernels.
    - [x] Non-stellsym asymmetric tables from `wout_10x10.nc` replay
      the CPU spline-list values for `rmns`, `zmnc`, `lmnc`, `gmns`,
      `bmns`, `bsup*ns`, and `bsub*ns` families.
  - [x] `tests/mhd/test_vmec_compute_geometry_jax.py`:
    - [x] Identity-with-CPU on
      `tests/test_files/wout_li383_low_res_reference.nc` for two `s`
      values × `(ntheta, nphi)` = `(4, 5)` across all
      `VmecGeometryResults` array fields. Tolerance: same-machine
      `rtol=1e-10`, `atol=1e-12`.
    - [x] CPU `vmec_fieldlines(..., theta1d=...)` theta/phi tensors
      can be passed directly to `vmec_compute_geometry_jax`; the
      fieldline-consumed gyrokinetic fields match CPU.
    - [x] First-derivative parity via `jax.grad` vs centered
      finite difference (`derivative_heavy` lane) for a frozen
      `bmnc` spline coefficient flowing through spline evaluation,
      Fourier reconstruction, and `modB`.
    - [x] Stellsym-on and stellsym-off branches each get coverage,
      including non-stellsym `lmnc`/asymmetric field families from
      `wout_10x10.nc`.
- [x] **N3.6 — Acceptance**
  - [x] N3.1/N3.2 evidence: `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
    pytest -q -p no:cacheprovider tests/mhd/test_vmec_frozen.py` passed
    on 2026-05-19 (`5 passed`).
  - [x] Local quality evidence for the freeze slice: `ruff check`,
    `ruff format --check`, and isolated
    `mypy --cache-dir=/dev/null --follow-imports=skip --ignore-missing-imports`
    passed on `src/simsopt/mhd/_vmec_frozen.py` and
    `tests/mhd/test_vmec_frozen.py` on 2026-05-19.
  - [x] N3.3/N3.4 focused CPU evidence: `JAX_ENABLE_X64=True
    JAX_PLATFORM_NAME=cpu pytest -q -p no:cacheprovider
    tests/mhd/test_vmec_compute_geometry_jax.py` passed on
    2026-05-19 (`9 passed`).
  - [x] Local quality evidence for the N3.3/N3.4 geometry slice:
    `ruff check --no-cache`, `ruff format --check --no-cache`,
    isolated
    `mypy --cache-dir=/dev/null --follow-imports=skip --ignore-missing-imports`,
    and `git diff --check` passed on
    `src/simsopt/jax_core/vmec_geometry.py` and
    `src/simsopt/mhd/vmec_diagnostics_jax.py`,
    `src/simsopt/mhd/__init__.py`, and
    `tests/mhd/test_vmec_compute_geometry_jax.py` on 2026-05-19.
  - [x] All N3.5 tests pass on CPU.
  - [x] N3.5 subset runs on CUDA.
    Perlmutter debug job `53204536` (`ljax-mhd`) requested
    `--gpus-per-node=1` under `-q debug -C gpu`, ran with
    `JAX_PLATFORMS=cuda,cpu` and
    `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`, and passed
    `tests/mhd/test_vmec_compute_geometry_jax.py` as part of the MHD CUDA
    smoke (`60 passed in 181.65s`) on 2026-05-20.
  - [x] `transfer_guard("disallow")` sweep over the inner loop quiet
    for the focused N3.3 JIT test.
  - [x] Bench note recorded in
    `docs/vmec_compute_geometry_jax_bench_2026-05-20.md` with StableHLO
    node count (`6236`) and one-call CUDA wall time
    (`0.001767153007676825` s). Artifact: Perlmutter debug job
    `53204761` (`ljax-n3bench`), `jax==0.10.0`, CUDA backend.

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

- [x] **N4.1 — Newton kernel**
  - [x] Add `simsopt.jax_core/_root.py` as the shared Newton SSOT from
    §2.3. It exports `newton_scan_fixed_iters(...)` for
    differentiating through a fixed `jax.lax.scan` iteration contract
    and `newton_with_implicit_vjp(...)` for converged root solves with
    the IFT adjoint
    $(\partial r / \partial x)^T \lambda = \bar{x}$ and
    $-(\partial r / \partial p)^T \lambda$ parameter cotangents.
    Evidence: `tests/jax_core/test_root.py`.
  - [x] Re-use `simsopt.jax_core/_root.py` in
    `simsopt.jax_core/vmec_fieldlines.py` with the residual

      $$ r(\theta_v) = \theta_{p,\text{target}}
        - (\theta_v + \sum_{mn} \lambda_{mn}(s)
        \sin(m\theta_v - n\phi)) $$

    Evidence:
    `tests/mhd/test_vmec_fieldlines_jax.py::test_theta_vmec_from_theta_pest_scan_matches_cpu_fieldlines_theta_branch`
    and
    `tests/mhd/test_vmec_fieldlines_jax.py::test_theta_vmec_from_theta_pest_implicit_matches_cpu_fieldlines_phi_branch`.
  - [x] `jax.vmap` over `(ns, nalpha)` while keeping `nl` as the vector
    unknown inside each per-line Newton solve, returning the full
    `(ns, nalpha, nl)` theta tensor from one staged call. Use
    `newton_with_implicit_vjp` for the reverse-mode contract; use
    `newton_scan_fixed_iters` only for a separately tested
    finite-iteration contract. Evidence:
    `tests/mhd/test_vmec_fieldlines_jax.py::test_theta_vmec_implicit_gradient_matches_finite_difference`
    and
    `tests/mhd/test_vmec_fieldlines_jax.py::test_theta_vmec_implicit_transfer_guard_clean`.
  - [x] Root-kernel evidence: `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu
    pytest -q -p no:cacheprovider tests/jax_core/test_root.py` passed
    on 2026-05-19 (`5 passed`), including explicit-Jacobian solve,
    implicit scalar/vector parameter gradients, `jax.jit`, and
    transfer-guard coverage.
  - [x] Fieldline theta-kernel evidence: `JAX_ENABLE_X64=True
    JAX_PLATFORM_NAME=cpu pytest -q -p no:cacheprovider
    tests/mhd/test_vmec_fieldlines_jax.py` passed on 2026-05-19
    (`10 passed`), including CPU `vmec_fieldlines` theta/phi branch
    parity, residual closure, explicit warm-start seed wiring,
    branch-stable warm-start resolution, implicit FD gradient,
    public wrapper parity, and transfer guard.
- [x] **N4.2 — Public wrapper**
  - [x] `simsopt.mhd/vmec_diagnostics_jax.py` adds
    `vmec_fieldlines_jax(vs, s, alpha, theta1d=None, phi1d=None, phi_center=0.0)`.
    Returns the same dataclass as N3 with the field-line-specific
    attributes `nalpha`, `nl`, `alpha`, `theta1d`, `phi1d` appended.
    Evidence:
    `tests/mhd/test_vmec_fieldlines_jax.py::test_public_vmec_fieldlines_jax_theta_branch_matches_cpu`,
    `tests/mhd/test_vmec_fieldlines_jax.py::test_public_vmec_fieldlines_jax_phi_branch_matches_cpu`,
    and
    `tests/mhd/test_vmec_fieldlines_jax.py::test_public_vmec_fieldlines_jax_accepts_vmec_object_and_lazy_mhd_export`.
- [x] **N4.3 — Tests**
  - [x] `tests/mhd/test_vmec_fieldlines_jax.py`:
    - [x] Identity-with-CPU at the same `(s, alpha, theta1d)` /
      `(s, alpha, phi1d)` for both branches; tolerance set by the
      inner-Newton residual (`<=1e-12`) on top of N3’s parity bound.
    - [x] `branch_stable_resolve` lane: explicit warm-start seed
      wiring is covered by a zero-iteration fixed-scan check, and
      re-solve with two warm-starts gives identical theta_vmec up
      to numerical noise.
    - [x] `derivative_heavy` lane: `jax.grad` of
      `||grad_psi_dot_grad_psi||_2` wrt frozen state matches
      centred FD on the same fixture. Evidence:
      `tests/mhd/test_vmec_fieldlines_jax.py::test_public_vmec_fieldlines_jax_frozen_state_gradient_matches_finite_difference`.
- [x] **N4.4 — Acceptance**
  - [x] N4.2 public-wrapper CPU evidence: `JAX_ENABLE_X64=True
    JAX_PLATFORM_NAME=cpu pytest -q -p no:cacheprovider
    tests/mhd/test_vmec_fieldlines_jax.py` passed on 2026-05-19
    (`10 passed`).
  - [x] N4.2/N4.3 local quality evidence: `ruff check --no-cache`,
    `ruff format --check --no-cache`, isolated
    `mypy --cache-dir=/dev/null --follow-imports=skip --ignore-missing-imports`,
    and `git diff --check` passed on
    `src/simsopt/jax_core/vmec_fieldlines.py`,
    `src/simsopt/mhd/vmec_diagnostics_jax.py`, and
    `tests/mhd/test_vmec_fieldlines_jax.py` on 2026-05-19.
  - [x] Full N4.3 passes on CPU.
  - [x] N4.3 smoke on CUDA.
    Perlmutter debug job `53204723` (`ljax-n3n4`) requested
    `--gpus-per-node=1` under `-q debug -C gpu`, ran with
    `JAX_PLATFORMS=cuda,cpu` and
    `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`, and passed
    `tests/jax_core/test_root.py` on CUDA on 2026-05-20
    (`6 passed in 9.53s`). The later benchmark step in the same Slurm
    packet failed due a bad fixture path and was rerun successfully as
    job `53204761`; it did not affect the completed N4 pytest smoke.
  - [x] No host pulls in the inner Newton in the existing transfer-guard
    test; the only allowed
    materialisation is at the public boundary.

---

## 7. Milestone N5 — QFM Surface orchestration

### 7.1 Rationale and purpose

`QfmResidualJAX` already exists in `simsopt.geo.surfaceobjectives_jax`
alongside the `surface_qfm_*_jax_from_dofs` helpers. The remaining gap is the
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

- [x] **N5.1 — `simsopt.jax_core/qfm_solver.py`**
  - [x] Pure functions
    `qfm_penalty_solve_jax(spec, coil_set_spec, label, targetlabel,
    constraint_weight, init_dofs, *, label_spec, label_coil_set_spec,
    max_iter, tol, optimizer)`
    returning `(final_dofs, info_pytree)` are implemented for
    `optimizer="bfgs"` via an in-repo fixed-iteration BFGS loop whose
    line-search and inverse-Hessian state stay in staged JAX arrays.
    The existing
    `surface_qfm_*_jax_from_dofs` functions now delegate to
    `simsopt.jax_core.qfm_solver` as the SSOT. Optional `"lm"` /
    `"optimistix-bfgs"` routes remain unwired and fail closed instead
    of falling back.
  - [x] `qfm_augmented_lagrangian_solve_jax(spec, coil_set_spec,
    label, targetlabel, init_dofs, *, label_spec, label_coil_set_spec,
    max_outer, inner_max_iter, tol)`
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
- [x] **N5.3 — Tests**
  - [x] `tests/geo/test_qfmsurface_jax.py`:
    - [x] Public JAX BFGS penalty solve reduces the fixed-state
      QFM penalty on a low-resolution NCSX fixture.
    - [x] `QfmSurfaceJAX.qfm_penalty_constraints(..., derivatives=1)`
      matches the pure value/gradient helper without mutating
      `surface.x`.
    - [x] `QfmSurfaceJAX` preserves the CPU `ToroidalFlux` contract where
      the label can own a different `BiotSavart` object from the QFM
      residual field. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_surface_jax_toroidal_flux_uses_label_owned_biotsavart`.
    - [x] The in-repo BFGS solver core runs under
      `jax.transfer_guard("disallow")` on the low-resolution NCSX
      fixture.
    - [x] The augmented-Lagrangian wrapper keeps scalar multiplier /
      penalty-weight updates and the inner BFGS call clean under
      `jax.transfer_guard("disallow")` for a two-outer-step smoke that
      exercises the multiplier and penalty-weight update.
    - [x] The augmented-Lagrangian result schema pairs public
      `fun=QFM residual` with the QFM objective gradient rather than
      the augmented-objective gradient.
    - [x] The augmented-Lagrangian diagnostics report `fun=QFM residual`,
      `augmented_value`, `multiplier`, and `penalty_weight` for the
      final inner objective actually minimized, not the next outer-loop
      state.
    - [x] Strict JAX backend dispatch writes final DOFs only after the
      pure solve and does not enter native SLSQP for `method="AL"`.
    - [x] Native dispatch rejects the unwired JAX-only `method="LM"`
      instead of silently routing it to host SLSQP.
    - [x] `simsopt.geo` lazy-exports `QfmSurfaceJAX`.
    - [x] Penalty path diagnostic produces residual ≤ host-SciPy LBFGS-B
      path on the `tests/geo/test_qfm.py` SurfaceXYZFourier volume
      fixture. This is a solver-quality diagnostic, not same-state
      fixed-point parity. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_penalty_solve_jax_not_worse_than_host_lbfgsb_diagnostic`.
    - [x] AL path satisfies equality residual $|L - L_{\text{target}}| \le 10^{-6}$
      on the same fixture after the upstream host LBFGS warm start and
      reaches QFM residual no worse than the host SLSQP exact-path
      diagnostic. The earlier
      proposed host-SLSQP DOF-identity check is not a valid acceptance
      contract: upstream SLSQP constrains `0.5 * (L - target)^2 == 0`,
      accepts label residuals at the `3e-5` fixture tolerance, and does
      not define a unique branch-stable DOF oracle for a stricter
      residual-constrained AL solve. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_meets_upstream_exact_acceptance`.
    - [x] AL success is now defined by the absolute natural-equality
      KKT residual for `label(dofs) = targetlabel`, not by the raw
      final inner BFGS status. The raw inner status remains in
      `QfmAugmentedLagrangianInfo.status` for diagnostics. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_success_uses_absolute_kkt`
      and
      `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_rejects_feasible_nonstationary_state`.
    - [x] AL branch stability is accepted by objective, label, and
      KKT invariants across small warm-start perturbations, not by
      host-SLSQP DOF identity. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_branch_stability_uses_kkt_invariants`.
    - [x] The volume fixture retains a relative host-SLSQP KKT
      diagnostic using the natural equality residual and projected
      scalar multiplier. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_kkt_diagnostic_no_worse_than_host_slsqp`.
    - [x] `derivative_heavy` lane: fixed-state penalty objective
      gradients match FD. For converged-solve sensitivities, test only
      the Optimistix / implicit-adjoint route; do not claim
      differentiation through the current forward BFGS solve. Evidence:
      `tests/geo/test_qfmsurface_jax.py::test_qfm_penalty_fixed_state_gradient_matches_centered_fd`.
- [x] **N5.4 — Acceptance**
  - [x] Focused CPU evidence before the KKT-success root fix:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q -p
    no:cacheprovider tests/geo/test_qfmsurface_jax.py` passed on
    2026-05-19 (`19 passed`).
  - [x] KKT-success root-fix evidence: `JAX_ENABLE_X64=True
    JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p
    no:cacheprovider
    tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_success_uses_absolute_kkt
    tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_rejects_feasible_nonstationary_state
    tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_branch_stability_uses_kkt_invariants`
    passed on 2026-05-19 (`3 passed`).
  - [x] Existing QFM penalty kernel evidence stayed green:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q -p
    no:cacheprovider tests/geo/test_surface_objectives_jax.py::TestQfmPenaltyJAX
    tests/geo/test_qfm.py::QfmSurfaceTests::test_qfm_penalty_constraints_gradient`
    passed on 2026-05-19 (`5 passed, 8 subtests passed`).
  - [x] Local quality evidence: `ruff check`, `ruff format --check`,
    and `git diff --check` passed on the N5 touched files on 2026-05-19.
    Isolated mypy was not run in the repo-local JAX environment because
    `mypy` is not installed there.
  - [x] No `surface.x = x` mutations inside the inner solve for the
    implemented BFGS/AL adapter seam; tests assert the surface is
    unchanged while the pure solver is running and only receives final
    device DOFs after the solver returns.
  - [x] Strict transfer-guard evidence: the implemented BFGS solver
    and AL wrapper paths are covered by
    `tests/geo/test_qfmsurface_jax.py::test_qfm_penalty_solve_jax_transfer_guard_clean`
    and
    `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_solve_jax_transfer_guard_clean`.
  - [x] Host-SciPy LBFGS-B residual diagnostic is covered by
    `tests/geo/test_qfmsurface_jax.py::test_qfm_penalty_solve_jax_not_worse_than_host_lbfgsb_diagnostic`.
  - [x] Host SLSQP exact-path acceptance is covered as a CPU diagnostic by
    `tests/geo/test_qfmsurface_jax.py::test_qfm_augmented_lagrangian_meets_upstream_exact_acceptance`;
    the test proves the host SLSQP run meets the upstream objective and
    label-residual bounds, and separately proves the JAX AL run meets
    the stricter label-residual bound with QFM residual no worse than
    the host SLSQP diagnostic.
  - [x] Host SLSQP DOF/solution identity is intentionally not claimed
    for the stricter AL equality contract above. Branch stability is
    closed by non-degenerate KKT/objective/label invariants instead,
    because upstream host SLSQP constrains `0.5 * residual**2 == 0`
    and is degenerate near feasibility.
  - [x] CUDA smoke passed on Perlmutter.
    Debug job `53204537` (`ljax-qfm`) requested `--gpus-per-node=1`
    under `-q debug -C gpu`, ran with `JAX_PLATFORMS=cuda,cpu` and
    `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`, and passed
    the QFM CUDA smoke on 2026-05-20 (`23 passed in 525.87s`).

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
  the device for the inner step through the optimizer runtime
  dependencies in `simsopt[JAX]` / `simsopt[JAX_GPU]`;
- replaces the per-DOF finite-difference Jacobian with `jax.jacfwd`
  in the gradient-on lane and with a `shard_map`-parallel forward
  finite-difference pass in the no-gradient lane on a single host
  with multiple GPUs;
- keeps the MPI worker partition for multi-host scenarios. The MPI
  outer loop still owns DOF distribution, but the *inner* gradient
  evaluation can now reside on each rank’s local GPU via JAX.

### 8.2 Detailed implementation plan

- [x] **N6.1 — `simsopt.solve/serial_jax.py`**
  - [x] `least_squares_serial_solve_jax(prob, *, optimizer="lm", rtol=..., atol=..., max_steps=..., **kwargs)`
    provides the traceable JAX least-squares lane corresponding to
    `least_squares_serial_solve`.
    Require the traceable-problem adapter explicitly; do not wrap
    arbitrary host `Optimizable` graphs or copy the host driver's
    `try`/large-residual recovery behavior. Implemented for explicit
    `TraceableLeastSquaresProblem` state with Optimistix
    `LevenbergMarquardt` / `GaussNewton`; arbitrary host graph inputs
    fail at the public contract boundary instead of being wrapped. The
    lane is exported through `simsopt.solve` while keeping the Optimistix
    import at solve-construction time, so importing `simsopt.solve` does
    not require the optional optimizer package until the JAX lane is used.
  - [x] Iteration log uses the same `simsopt_<datestr>.dat` header and
    row layout as the host driver, with full `function_evaluation`
    semantics covered by `tests/solve/test_serial_jax.py`: least-squares,
    scalar, and constrained lanes now assert multiple ordered rows whose
    `function_evaluation` values equal `range(len(rows))` rather than only
    header/final-row compatibility.
  - [x] `serial_solve_jax` and `constrained_serial_solve_jax` follow
    the same explicit traceable-problem template. `serial_solve_jax`
    accepts `TraceableScalarProblem` and drives Optimistix BFGS on the
    scalar objective; `constrained_serial_solve_jax` accepts
    `TraceableEqualityConstrainedProblem` and runs the same
    Hestenes–Powell augmented-Lagrangian update policy used by N5 with
    Optimistix BFGS inner steps. Neither wrapper infers traceability
    from arbitrary host graphs or falls back to host SLSQP. Evidence:
    `tests/solve/test_serial_jax.py::test_serial_solve_jax_matches_host_general_quadratic_problem`
    and
    `tests/solve/test_serial_jax.py::test_constrained_serial_solve_jax_matches_host_slsqp_equality_problem`.
- [x] **N6.2 — Finite-difference Jacobian via `shard_map`**
  - [x] `simsopt.jax_core/_finite_difference.py` adds
    `forward_jacobian_shard_map(fn, x0, abs_step, rel_step,
    diff_method, mesh)` that distributes the per-DOF perturbations
    across a `shard_map` over an `('dof',)` mesh axis. `mesh` is
    explicit. The single-device sibling is
    `forward_jacobian_vmap(...)`; callers choose one route by backend
    policy rather than relying on implicit route selection. CPU
    single-device mesh and fake two-device transfer-guard evidence:
    `tests/jax_core/test_finite_difference.py`. Step sizing follows
    the existing SIMSOPT finite-difference SSOT
    `max(abs(x) * rel_step, abs_step)` and rejects zero steps.
  - [x] When `JAX_PLATFORM_NAME=cuda` and `XLA_FLAGS` enables
    real GPU execution, devices are read from `jax.devices()`. CPU
    fake-device tests may use `--xla_force_host_platform_device_count`
    before JAX import, but that is test-only. Static-shape constraint:
    pad the DOF axis to the mesh size inside the explicit sharded
    function and discard padded columns at assembly. Perlmutter debug
    job `53204538` (`ljax-n6`) requested `--gpus-per-node=4`, ran with
    `JAX_PLATFORMS=cuda,cpu` and
    `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`, and passed
    the N6 CUDA/mode-matrix smoke on 2026-05-20
    (`24 passed, 3 skipped in 37.40s`). Dedicated real-GPU shard-map
    proof job `53204864` (`ljax-n6multi`) then built a
    `Mesh(np.asarray(jax.devices("cuda")[:4]), ("dof",))` over four
    CUDA devices, used a 7-DOF linear residual to exercise padding
    (`dof_count_mod_device_count = 3`), ran under
    `jax.transfer_guard("disallow")`, and matched the analytic Jacobian
    with `max_abs_error = 0.0`.
- [x] **N6.3 — `simsopt.solve/mpi_jax.py`**
  - [x] `least_squares_mpi_solve_jax(prob, mpi, *, …)` reuses the
    existing `MpiPartition` worker pool but offloads each group
    leader's Jacobian column block to the JAX path of N6.2 — i.e. each
    group leader runs `forward_jacobian_shard_map` over its assigned
    columns locally, then assembles the Jacobian through the leaders
    communicator. Evidence:
    `tests/solve/test_mpi_jax.py::test_traceable_least_squares_mpi_jacobian_matches_jacfwd`.
  - [x] `mpi.comm_groups.bcast` of `x` stays exactly as in the host
    driver for the Jacobian lane, and `least_squares_mpi_solve_jax`
    solves on rank 0 with the MPI/JAX Jacobian callback, then broadcasts
    the final state with `mpi.comm_world.Bcast`.
    Evidence:
    `tests/solve/test_mpi_jax.py::test_least_squares_mpi_solve_jax_reaches_traceable_quadratic_optimum`.
- [x] **N6.4 — Tests**
  - [x] `tests/solve/test_serial_jax.py`:
    - [x] Identity-with-host on the existing
      `tests/objectives/test_least_squares.py` toy problem with
      a `TraceableLeastSquaresProblem` adapter over the JAX-traced
      residual kernel. The test asserts both host SciPy and JAX
      Optimistix lanes reach the analytic toy optimum and near-zero
      objective; it does not claim bit-identical solver iterates.
    - [x] `derivative_heavy` lane: `jax.jacfwd` Jacobian matches
      `forward_jacobian_shard_map` output within `rtol=1e-10`.
    - [x] Arbitrary host `LeastSquaresProblem` inputs are rejected by
      the JAX lane instead of being implicitly wrapped.
    - [x] `serial_solve_jax` matches the host general-solve optimum on
      an independent quadratic objective while preserving the host
      `simsopt_<datestr>.dat` general-problem header/layout.
    - [x] `constrained_serial_solve_jax` matches host SLSQP and the
      analytic equality-constrained quadratic optimum using the
      traceable augmented-Lagrangian lane, not host fallback.
    - [x] Arbitrary host scalar / constrained problem inputs are
      rejected by the JAX scalar and constrained lanes instead of being
      implicitly wrapped.
    - [x] `simsopt.solve` exports the traceable JAX serial,
      least-squares serial, and constrained serial lanes.
  - [x] `tests/solve/test_mpi_jax.py` (gated behind `mpi4py` import):
    - [x] 2-rank smoke run reproducing the traceable serial JAX solver
      state to `rtol=1e-12` on the same seed. Host `LeastSquaresProblem`
      graphs remain out of scope for the explicit traceable-problem
      protocol.
    - [x] MPI leader-owned JAX finite-difference column blocks assemble
      the same Jacobian as `jax.jacfwd` at `rtol=1e-10`.
  - [x] `tests/jax_core/test_finite_difference.py`:
    - [x] `forward_jacobian_vmap` matches `jax.jacfwd` on a linear
      residual and a centered nonlinear finite-difference oracle.
    - [x] `forward_jacobian_vmap` uses the same absolute/relative step
      contract as `simsopt._core.util.finite_difference_steps` and
      rejects zero finite-difference steps, including materialized
      low-precision zero steps, instead of returning NaNs. Compiled
      `jax.jit`/`shard_map` routes fail closed when `abs_step`
      materializes to zero because traced coordinates cannot be inspected
      for the CPU helper's data-dependent zero-coordinate exception.
    - [x] `forward_jacobian_shard_map` matches the vmap route on an
      explicit one-device `('dof',)` mesh and is quiet under
      `jax.transfer_guard("disallow")`.
    - [x] `forward_jacobian_shard_map` runs in a fresh fake-two-CPU-device
      subprocess with mesh-replicated inputs and closed-over residual data,
      proving the multi-device route has no implicit mesh transfer under
      `jax.transfer_guard("disallow")`.
- [x] **N6.5 — Acceptance**
  - [x] CPU pass on N6.4; CUDA smoke at 1 device.
    Partial CPU evidence for the N6.2 kernel slice:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q -p
    no:cacheprovider tests/jax_core/test_finite_difference.py`
    passed on 2026-05-19 (`10 passed`), including a fresh-process
    fake-two-CPU-device transfer-guard test.
    Partial CPU evidence for the N6.1/N6.2 serial/Jacobian slice:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu pytest -q -p
    no:cacheprovider tests/solve/test_serial_jax.py
    tests/jax_core/test_finite_difference.py` passed again on
    2026-05-19 after the general/constrained serial JAX additions
    (`17 passed`).
    Partial CPU evidence for the N6.3 MPI JAX slice:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python
    -m pytest -q -p no:cacheprovider tests/solve/test_mpi_jax.py`
    passed on 2026-05-19 (`2 passed`). Earlier two-rank
    `mpiexec -n 2` evidence also passed on 2026-05-19
    (`2 passed` per rank).
    Combined N6 CPU evidence:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python
    -m pytest -q -p no:cacheprovider tests/solve/test_serial_jax.py
    tests/jax_core/test_finite_difference.py tests/solve/test_mpi_jax.py`
    passed on 2026-05-19 (`19 passed`).
  - [x] Multi-GPU smoke for N6.2 is **hardware-gated** (cross-link
    against N30 in `docs/jax_native_round3_curated_todos_2026-05-18.md`).
    Hardware proof was recorded on Perlmutter debug job `53204864`
    (`ljax-n6multi`), which requested `--gpus-per-node=4`, constructed a
    four-CUDA-device `('dof',)` mesh from `jax.devices("cuda")`, and
    proved the padded finite-difference shard-map path exactly
    (`max_abs_error = 0.0`) on 2026-05-20.

---

## 9. Milestone N7 — Live PM / wireframe workflows

### 9.1 Rationale and purpose

`simsopt.solve/permanent_magnet_optimization_jax.py`,
`simsopt.solve/wireframe_optimization_jax.py`, and the
`simsopt.jax_core.pm_optimization` / `wireframe` kernels already carry
substantial `jax.lax.scan` coverage for the inner maths. The remaining
workflow boundary is now explicit: output writers
(FAMUS files, VTK, JSON, plots) and host callback cadence are host-side,
while restart snapshots, pruning, and final-adjustment state changes are
represented by JAX workflow loops. The audit
explicitly does not call for byte-identity output writers; it asks that
the **decision logic that changes the numerical state** runs in a
single explicit JAX loop, with host I/O only at documented boundaries.

### 9.2 Detailed implementation plan

- [x] **N7.1 — `simsopt.jax_core/pm_workflow.py`**
  - [x] PM GPMO live-loop adapters wrap the per-step `GPMO_*`
    kernels in fixed-length `jax.lax.scan` calls with active/done
    carries. The state pytrees carry the moment matrix, active-loop
    mask, and fixed-capacity history slices needed for restart across
    baseline, multi-neighbour, ArbVec, backtracking, and
    ArbVec-backtracking variants.
  - [x] Baseline-GPMO pruning rules are expressed as pure functions
    `(state) -> (state, prune_mask)` so the loop body stays
    JIT-friendly.
  - [x] 2026-05-19 local N7 seed: `src/simsopt/jax_core/pm_workflow.py`
    adds a restartable baseline-GPMO live-loop state
    (`PMGPMOLiveState`) with fixed-capacity selected-dipole,
    selected-component, selected-sign, and residual histories. The loop
    uses `jax.lax.scan`, an active/done carry, and pure
    `prune_rule` / `stop_rule` callables. This is a baseline-GPMO slice
    only.
  - [x] 2026-05-19 local N7 multi-neighbour slice:
    `src/simsopt/jax_core/pm_workflow.py` adds a restartable
    `PMGPMOMultiLiveState`, `pm_gpmo_multi_initial_state`, and
    `pm_gpmo_multi_live_loop_jax`. The loop wraps `gpmo_multi_step` in
    a fixed-length `jax.lax.scan`, carries selected seed dipoles,
    components, signs, residuals, and `Nadjacent` selected groups, and
    validates the existing multi-GPMO capacity rule
    `(state.steps_taken + max_steps) * Nadjacent <= ndipoles` before
    scan.
  - [x] 2026-05-19 local N7 ArbVec slice:
    `src/simsopt/jax_core/pm_workflow.py` adds a restartable
    `PMGPMOArbVecLiveState`, `pm_gpmo_arbvec_initial_state`, and
    `pm_gpmo_arbvec_live_loop_jax`. The loop wraps `gpmo_arbvec_step`
    in a fixed-length `jax.lax.scan`, carries selected dipoles, selected
    polarization-vector indices, signs, and residual history, and
    validates restart-expanded `K` through the existing
    `_validate_gpmo_arbvec_static_args` core-solver guard before scan.
  - [x] 2026-05-19 local N7 backtracking slice:
    `src/simsopt/jax_core/pm_workflow.py` adds a restartable
    `PMGPMOBacktrackingLiveState`, `pm_gpmo_backtracking_initial_state`,
    and `pm_gpmo_backtracking_live_loop_jax`. The loop wraps
    `gpmo_backtracking_step` in a fixed-length `jax.lax.scan`, carries
    the dewyrming state needed by future restart calls
    (`current_signs`, `current_components`, selected placement traces,
    residual history, `x_history`, nonzero counts, removed-pair counts,
    and done history), and validates restart-expanded `K` through the
    existing `_validate_gpmo_backtracking_static_args` core-solver guard
    before scan.
  - [x] 2026-05-19 local N7 ArbVec-backtracking slice:
    `src/simsopt/jax_core/pm_workflow.py` adds a restartable
    `PMGPMOArbVecBacktrackingLiveState`,
    `pm_gpmo_arbvec_backtracking_initial_state`, and
    `pm_gpmo_arbvec_backtracking_live_loop_jax`. The loop wraps
    `gpmo_arbvec_backtracking_step` in a fixed-length `jax.lax.scan`,
    carries the arbitrary-vector dewyrming state
    (`current_vector_indices`, `current_signs`, selected placement
    traces, residual history, `x_history`, nonzero counts,
    removed-pair counts, done history, and optional `x_init`
    initialization state), validates restart-expanded `K` through the
    existing `_validate_gpmo_arbvec_backtracking_static_args`
    core-solver guard before scan, and normalizes scan specs through the
    same casted payload contract used by the core PM solvers.
  - [x] The baseline live loop preserves the existing strict
    `K <= ndipoles`-style capacity contract: concrete restart state is
    rejected before scan when `state.steps_taken + max_steps` exceeds
    history or dipole capacity. All history arrays must share the same
    capacity, and traced restart counters are rejected so JAX scatter
    out-of-bounds behavior cannot silently mask a bad restart.
  - [x] `relax_and_split_jax` no longer runs its relax-and-split outer
    loop as a Python host loop. The L0/L1 relax-and-split path now
    validates the MwPGP `alpha` contract once at the host boundary,
    then executes a fixed-length `jax.lax.scan` whose carry holds
    `(m, m_proxy, done)`. Early convergence from
    `||m - prox(m)|| < epsilon_RS` updates the `done` carry and later
    scan slots carry the final state unchanged, so state-changing
    decision logic stays in the staged loop while preserving fixed
    output shapes.
- [x] **N7.2 — `simsopt.jax_core/wireframe_workflow.py`**
  - [x] `gsco_live_loop_jax(state, *, max_steps, params, stop_rule)`
    analogously wraps the existing GSCO inner step.
    `src/simsopt/jax_core/wireframe_workflow.py` now owns
    `WireframeGSCOLiveState`, `WireframeGSCOLiveParams`, and the
    fixed-shape GSCO scan. `src/simsopt/solve/wireframe_optimization_jax.py`
    imports the kernel from `jax_core` so the numerical loop has one
    source of truth; host matrix preparation, current mutation, and
    result dictionary assembly remain in the solve wrapper.
    Eager and staged `jax.jit` restart calls validate
    `history_length + max_steps` against history capacity before scan.
    The staged path uses `jax.experimental.io_callback` as a sequenced
    pre-scan contract check, not `jax.debug.callback`; this follows the
    official JAX callback contract where `io_callback` has guaranteed
    execution and `debug.callback` does not. The validated
    `history_length` returned by the callback is fed into the scan state,
    so JAX scatter out-of-bounds update semantics cannot silently mask a
    bad restart.
  - [x] Wireframe multistep final-adjustment orchestration from
    `examples/3_Advanced/wireframe_gsco_multistep.py` is represented by
    `wireframe_gsco_multistep_loop_jax`. The bounded outer
    `jax.lax.scan` carries the current vector, previous current vector,
    loop counts, enclosed-segment mask, current-fraction schedule, and
    final-adjustment status. Each active outer slot runs the fixed-state
    GSCO scan, removes sub-threshold coils by connected-component size,
    constrains zero-current segments enclosed by surviving coils, halves
    the current fraction, and then runs the final adjustment with
    `match_current=True` / `no_new_coils=True` once the pruned solution
    stops changing. Plotting and VTK writes remain host I/O boundaries.
- [x] **N7.3 — Host-side I/O boundary**
  - [x] `optimize_wireframe_jax` materialises fixed-state RCLS and
    GSCO result dataclasses via one final `jax.device_get` per solve
    result, then performs NumPy scalar/history shaping from that host
    copy. This follows the official JAX `device_get` pytree contract:
    pytree leaves are copied to host in parallel.
  - [x] No FAMUS / VTK / JSON / matplotlib output writers exist in the
    JAX PM or wireframe solve wrappers. The only solve-level write side
    effect in this slice is `optimize_wireframe_jax` mutating
    `wframe.currents` from the final host result after `_host_pytree`.
    Any future output writer should consume final-state host pytrees only,
    not add mid-loop `device_get` calls.
  - [x] A `record_every: Optional[int]` parameter optionally pulls
    snapshots out of the device tape at fixed cadence for
    downstream plotting; this materialisation cost is paid only
    when the user opts in.
    - [x] PM baseline-GPMO accepts `record_every` at the core
      `gpmo_baseline_solve` scan boundary. The default keeps existing
      full per-step histories. When supplied, the scan records only the
      requested cadence rows plus the final row for plotting/output
      history fields while leaving the final optimizer state unchanged.
      This avoids first constructing full `K`-row histories and then
      slicing them at the solve wrapper.
    - [x] PM multi-GPMO accepts `record_every` at the core
      `gpmo_multi_solve` scan boundary with the same bounded-history
      contract as baseline-GPMO. The default keeps existing full
      per-step histories. When supplied, the scan records only the
      requested cadence rows plus the final row while leaving the final
      optimizer state unchanged.
    - [x] PM ArbVec-GPMO accepts `record_every` at the core
      `gpmo_arbvec_solve` scan boundary. The default keeps existing
      full per-step histories. When supplied, the scan records only the
      requested cadence rows plus the final row while leaving the final
      optimizer state unchanged.
    - [x] PM backtracking and ArbVec-backtracking accept `record_every`
      at their core scan boundaries. The default keeps existing full
      per-step histories. When supplied, the scan records only the
      requested output-history rows plus the final row while leaving the
      final optimizer state unchanged. The backtracking algorithms still
      carry their `K`-length selected-placement state internally because
      the dewyrming pass consumes prior placements.
    - [x] Wireframe GSCO public-history cadence accepts `record_every`
      on the fixed-state public GSCO path. The default keeps existing
      full accepted histories. When supplied, the scan records the
      initial row, cadence accepted rows, and the final accepted row for
      plotting while preserving the final current vector and loop-count
      state. Sampled GSCO histories deliberately reject
      `get_gsco_iteration_jax` replay because skipped current updates
      cannot reconstruct dense intermediate states.
- [x] **N7.4 — Tests**
  - [x] `tests/solve/test_pm_workflow_jax.py`:
    - [x] Live-loop output equals step-by-step host loop output for
      a deterministic baseline PM fixture at `rtol=1e-12` on identical
      inputs.
    - [x] Restart from snapshot reproduces continuation step exactly.
    - [x] Pure pruning rule changes candidate selection before the
      baseline-GPMO placement step.
    - [x] Capacity-overrun, malformed-history-capacity, and
      traced-restart-counter tests cover the strict pre-scan guard.
    - [x] JIT + `jax.transfer_guard("disallow")` CPU smoke covers the
      precompiled baseline live loop after input staging. Per official
      JAX transfer-guard semantics, this is not a substitute for CUDA
      transfer proof.
    - [x] Multi-neighbour GPMO live-loop output equals a step-by-step
      `gpmo_multi_step` host loop, restart continuation is exact,
      history/group-capacity violations are rejected before scan, traced
      restart counters are rejected, invalid `Nadjacent` and
      `single_direction` static arguments preserve the core
      `gpmo_multi_solve` guards, and the compiled path is quiet under
      `jax.transfer_guard("disallow")` after input staging.
    - [x] ArbVec GPMO live-loop output equals a step-by-step
      `gpmo_arbvec_step` host loop, restart continuation is exact,
      history-capacity violations are rejected before scan, traced
      restart counters are rejected, invalid polarization-vector shapes
      preserve the core `gpmo_arbvec_solve` guards, and the compiled
      path is quiet under `jax.transfer_guard("disallow")` after input
      staging.
    - [x] Backtracking GPMO live-loop output equals a step-by-step
      `gpmo_backtracking_step` host loop including dewyrming trace
      fields, restart continuation is exact, history-capacity violations
      are rejected before scan, traced restart counters are rejected,
      invalid `single_direction` and `backtracking` static arguments
      preserve the core `gpmo_backtracking_solve` guards, and the
      compiled path is quiet under `jax.transfer_guard("disallow")`
      after input staging.
    - [x] ArbVec-backtracking GPMO live-loop output equals a
      step-by-step `gpmo_arbvec_backtracking_step` host loop including
      arbitrary-vector dewyrming trace fields and optional `x_init`
      initialization state, restart continuation is exact,
      history-capacity violations are rejected before scan, traced
      restart counters are rejected, invalid polarization-vector shapes
      and non-Python `thresh_angle` values preserve the core
      `gpmo_arbvec_backtracking_solve` guards, and the compiled path is
      quiet under `jax.transfer_guard("disallow")` after input staging.
  - [x] `tests/solve/test_permanent_magnet_optimization_jax_item28.py`:
    - [x] Relax-and-split L0/L1 outer-loop output still matches the
      CPU wrapper for the multi-outer-step fixture.
    - [x] Explicit-`alpha` relax-and-split validation stays eager while
      the scanned body uses the already-validated value.
    - [x] High-`epsilon_RS` early convergence is handled inside the
      fixed-length `lax.scan`: later output slots repeat the final
      `m`/`m_proxy` state instead of executing more host iterations.
    - [x] `record_every` on all PM GPMO variants keeps only the
      requested device-output-history rows plus the final row inside the
      core scan, preserves the final optimizer state, rejects invalid
      cadence values, and stays JIT/transfer-guard clean for the
      compiled path.
  - [x] `tests/solve/test_wireframe_workflow_jax.py`:
    - [x] GSCO live loop reproduces the host loop on the existing
      fixture for `max_steps ∈ {5, 50}`.
    - [x] Restart continuation reproduces a single uninterrupted live
      loop exactly.
    - [x] The exposed state-argument live-loop API is JIT-callable.
    - [x] Eager and staged restart capacity overruns are rejected before
      scan; the staged bad-restart test fails through the pre-scan
      `io_callback` before any GSCO history scatter executes.
    - [x] JIT + `jax.transfer_guard("disallow")` CPU smoke covers the
      precompiled GSCO live loop after input staging. Per official JAX
      transfer-guard semantics, this is not a substitute for CUDA
      transfer proof.
    - [x] Wireframe multistep orchestration tests cover connected
      saddle-coil size detection, parity against a host reference that
      repeats C++ `GSCO` calls with the example's pruning/final-adjustment
      state machine, JAXPR evidence for staged `scan`, and the
      enclosed-zero-current segment mask used to constrain later
      non-final steps.
  - [x] `tests/solve/test_wireframe_optimization_jax_item31.py`:
    - [x] Public RCLS and GSCO wrappers each materialise the final
      registered result dataclass with exactly one `jax.device_get`
      call before host-side result assembly.
    - [x] Public GSCO `record_every` keeps the initial, cadence, and
      final plotting rows, preserves final currents/loop counts, rejects
      sampled-history dense replay, and stays JIT/transfer-guard clean
      for the compiled path.
- [x] **N7.5 — Acceptance**
  - [x] N7 baseline-PM tests pass on CPU:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_pm_workflow_jax.py`
    (`7 passed`).
  - [x] PM workflow tests pass on CPU after adding the multi-neighbour
    GPMO live-loop adapter:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_pm_workflow_jax.py -q`
    (`14 passed`).
  - [x] PM workflow tests pass on CPU after adding the ArbVec GPMO
    live-loop adapter:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_pm_workflow_jax.py -q`
    (`21 passed`).
  - [x] PM workflow tests pass on CPU after adding the backtracking
    GPMO live-loop adapter:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_pm_workflow_jax.py -q`
    (`28 passed`).
  - [x] PM workflow tests pass on CPU after adding the
    ArbVec-backtracking GPMO live-loop adapter and aligning the live
    loops with the core solvers' normalized scan-spec payloads:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_pm_workflow_jax.py -q`
    (`35 passed`).
  - [x] Existing PM wrapper regression remains green with the new
    baseline workflow module:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py -q`
    (`36 passed` before the relax-and-split scan refactor; re-run
    evidence below supersedes this count for item28).
  - [x] CUDA smoke passed on Perlmutter.
    Debug job `53204539` (`ljax-n7`) requested `--gpus-per-node=1`
    under `-q debug -C gpu`, ran with `JAX_PLATFORMS=cuda,cpu` and
    `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`, and passed
    the N7 PM/wireframe CUDA smoke on 2026-05-20 (`48 passed in 42.10s`).
  - [x] Source inspection shows no mid-loop `device_get` calls in the
    current baseline PM and wireframe GSCO live-loop scan bodies:
    `src/simsopt/jax_core/pm_workflow.py` has one concrete pre-scan
    restart-capacity `device_get`, and
    `src/simsopt/jax_core/wireframe_workflow.py` has no `device_get`.
    Wireframe's staged-capacity contract deliberately uses one pre-scan
    `io_callback` for invalid staged restart rejection. CPU
    transfer-guard smokes cover the precompiled paths after input
    staging.
  - [x] CUDA transfer proof for the N7 live loops is covered by the
    Perlmutter CUDA smoke above, including the transfer-guard tests in
    `tests/solve/test_pm_workflow_jax.py`,
    `tests/solve/test_wireframe_workflow_jax.py`, and
    `tests/solve/test_wireframe_optimization_jax_item31.py`.
  - [x] N7 wireframe GSCO tests pass on CPU:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_workflow_jax.py`
    (`7 passed`).
  - [x] Existing fixed-state wireframe regression remains green after
    moving GSCO loop ownership into `jax_core`:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_workflow_jax.py tests/solve/test_wireframe_optimization_jax_item31.py`
    (`39 passed`).
  - [x] Fixed-state wireframe wrapper regression remains green after
    collapsing final RCLS/GSCO result materialisation to one
    `device_get` per solve result:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py -q`
    (`34 passed`).
  - [x] N7 focused CPU sweep remains green after the host-boundary
    update:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`78 passed` before the PM GPMO `record_every` cadence; re-run
    evidence below supersedes this count for N7).
  - [x] PM solve wrapper regression remains green after replacing the
    relax-and-split Python outer loop with a fixed-length `lax.scan`:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_permanent_magnet_optimization_jax_item28.py -q`
    (`30 passed` before the PM GPMO `record_every` cadence; re-run
    evidence below supersedes this count for item28).
  - [x] PM solve wrapper regression remains green after adding
    core-scan `record_every` to PM baseline-GPMO:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_permanent_magnet_optimization_jax_item28.py -q`
    (`33 passed`).
  - [x] N7 focused CPU sweep remains green after adding the PM GPMO
    `record_every` cadence:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`81 passed`).
  - [x] PM solve wrapper regression remains green after extending
    core-scan `record_every` to PM multi-GPMO:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_permanent_magnet_optimization_jax_item28.py -q`
    (`35 passed`).
  - [x] N7 focused CPU sweep remains green after adding PM multi-GPMO
    `record_every`:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`83 passed`).
  - [x] PM solve wrapper regression remains green after extending
    core-scan `record_every` to PM ArbVec-GPMO:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_permanent_magnet_optimization_jax_item28.py -q`
    (`37 passed`).
  - [x] N7 focused CPU sweep remains green after adding PM ArbVec-GPMO
    `record_every`:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`85 passed`).
  - [x] PM solve wrapper regression remains green after extending
    core-scan `record_every` to PM backtracking and ArbVec-backtracking:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_permanent_magnet_optimization_jax_item28.py -q`
    (`41 passed`).
  - [x] N7 focused CPU sweep remains green after adding PM backtracking
    and ArbVec-backtracking `record_every`:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`89 passed`).
  - [x] Fixed-state wireframe wrapper regression remains green after
    adding GSCO `record_every` sampled plotting histories:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py -q`
    (`37 passed`).
  - [x] N7 focused CPU sweep remains green after adding wireframe GSCO
    `record_every`:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`92 passed`).
  - [x] N7 focused CPU sweep remains green after adding the
    multi-neighbour PM GPMO live-loop adapter:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`99 passed`).
  - [x] N7 focused CPU sweep remains green after adding the PM ArbVec
    GPMO live-loop adapter:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`106 passed`).
  - [x] N7 focused CPU sweep remains green after adding the PM
    backtracking GPMO live-loop adapter:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`113 passed`).
  - [x] N7 focused CPU sweep remains green after adding the PM
    ArbVec-backtracking GPMO live-loop adapter and normalized scan-spec
    payloads for PM live loops:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`120 passed`).
  - [x] Wireframe workflow tests pass on CPU after adding the
    multistep GSCO orchestration loop:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_workflow_jax.py -q`
    (`11 passed`).
  - [x] N7 focused CPU sweep remains green after adding the wireframe
    multistep GSCO orchestration loop:
    `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m pytest -q -p no:cacheprovider tests/solve/test_wireframe_optimization_jax_item31.py tests/solve/test_wireframe_workflow_jax.py tests/solve/test_pm_workflow_jax.py tests/solve/test_permanent_magnet_optimization_jax_item28.py`
    (`124 passed`).

---

## 10. Cross-Milestone Validation, CI, and Documentation

- [x] **V1 — Parity-ladder updates.** Each milestone cites an existing
  lane key from `benchmarks/validation_ladder_contract.py` in tests and
  closeout docs. Do not add or rename tolerance lanes unless a surface
  genuinely needs a new tolerance contract; if that happens, update the
  SSOT first and then update the table in §2.5.
  2026-05-20 update: §2.5 now covers N1–N7 using existing lane keys
  (`direct_kernel`, `derivative_heavy`, `branch_stable_resolve`,
  `ls_wrapper_gradient`, `fd_gradient`, `pm_mwpgp_fixed_step`, and
  `reporting_contract`). N6/N7 test modules cite the same lane keys in
  module docstrings or explicit tolerance imports from
  `benchmarks.validation_ladder_contract`.
- [x] **V2 — Strict gate sweep.** After every milestone merges, run
  `benchmarks/single_stage_init_parity.py` under `jax_cpu_parity` and
  `jax_gpu_parity` modes to confirm the byte-identity gate is intact.
  2026-05-20 update: strict CPU artifact
  `.artifacts/v2-strict-20260520-r22-cpu/single_stage_cpu.json` passed
  under `jax_cpu_parity`, `SIMSOPT_BACKEND_STRICT=1`, transfer guard
  `disallow`, JAX/JAXLIB `0.10.0`, x64 enabled, backend `cpu`, devices
  `['cpu:0']`, and failures `[]`; comparison deltas were
  `final_iota_abs_diff=0.0`, `final_volume_rel_diff=1.2501259683660253e-15`,
  `field_error_rel_diff=4.8921741994723286e-15`, and
  `max_surface_pointwise_rel=0.0`.
  Perlmutter debug job `53224316` wrote strict CUDA artifact
  `/pscratch/sd/j/jungdae/simsopt-jax-results/v2-strict-debug-20260520T111600Z-cuda-only-r20/single_stage_cuda.json`
  and completed `COMPLETED`/`0:0` in `00:21:30`; it passed under
  `jax_gpu_parity`, `SIMSOPT_BACKEND_STRICT=1`, transfer guard
  `disallow`, JAX/JAXLIB `0.10.0`, x64 enabled, backend `gpu`, devices
  `['cuda:0']`, `JAX_PLATFORMS=cuda,cpu`,
  `--xla_gpu_exclude_nondeterministic_ops=true`, and failures `[]`.
  CUDA comparison deltas were `final_iota_abs_diff=0.0`,
  `final_volume_rel_diff=8.334173122440179e-16`,
  `field_error_rel_diff=2.257926553602606e-15`, and
  `max_surface_pointwise_rel=0.0`.
- [x] **V3 — Mode-matrix smoke.** A new
  `tests/integration/test_remaining_jax_surfaces_mode_matrix.py`
  imports each new public entrypoint under all six
  `SIMSOPT_BACKEND_MODE` values and a `JAX_PLATFORM_NAME=cuda` smoke.
  2026-05-20 update: the test now imports the remaining simsoptpp-free
  MHD / solve / JAX-core surfaces under the SSOT
  `VALID_BACKEND_MODES` list, which is seven modes in the current tree
  (`native_cpu`, `jax_cpu_fast`, `jax_cpu_parity`,
  `jax_cpu_float32_smoke`, `jax_gpu_fast`, `jax_gpu_parity`,
  `jax_mps_smoke`). Local CPU evidence:
  `JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu .conda/jax/bin/python -m
  pytest -q -p no:cacheprovider
  tests/integration/test_remaining_jax_surfaces_mode_matrix.py`
  passed (`5 passed, 3 skipped`), with CUDA/MPS skipped because those
  platforms are not present locally. The top-level V3 checkbox is closed
  by the CUDA leg below.
  2026-05-20 CUDA evidence: Perlmutter debug job `53204984`
  (`ljax-v23`) ran the mode matrix with `JAX_PLATFORM_NAME=cuda`,
  `JAX_PLATFORMS=cuda,cpu`, and
  `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`; result:
  `7 passed, 1 skipped in 24.17s`. The single skip is the MPS smoke
  mode, which is not a CUDA target.
- [x] **V4 — CI hooks.** Update `.github/workflows/jax_smoke.yml` so
  the new pure-JAX modules and public wrappers that are documented as
  `simsoptpp`-free are imported under the smoke job. Add a CUDA-gated
  job under whatever runner we use for `jax_gpu_parity` smoke today.
  2026-05-20 update: `jax_smoke.yml` now triggers on `src/simsopt/mhd/**`,
  `src/simsopt/solve/**`, `tests/jax_core/**`, `tests/mhd/**`,
  `tests/solve/**`, and the new remaining-surface mode matrix; the CPU
  public smoke runs the mode matrix; the CUDA strict-purity job runs the
  same mode matrix under the existing self-hosted GPU runner; and stale
  `src/simsopt/backend.py` workflow references were removed in favor of
  the `simsopt.backend` package facade.
- [x] **V5 — Docs.**
  - [x] Extend `docs/source/jax_acceptance.rst` with the bootstrap /
    profiles / VMEC-diagnostics acceptance criteria.
  - [x] Add a “Frozen VMEC state” section to
    `docs/source/jax_gpu_setup.rst` covering N3.1.
  - [x] Cross-link this plan from `docs/jax_native_round3_curated_todos_2026-05-18.md`
    once the first milestone (N1) lands.

---

## 11. Risk Register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Frozen VMEC state diverges from live `Vmec.wout` if user mutates the run | medium | high | document frozen state as an immutable snapshot; users must call `vmec_freeze_splines` again after mutation |
| Augmented Lagrangian inner loop fails to converge on stiff equality constraint | medium | medium | expose `rho_max` and `beta` knobs and return a failed solver status for the selected JAX method |
| `shard_map` per-device Jacobian column block requires padding when `dof_size % n_devices != 0` | high | low | pad with no-op columns and discard at the rank-0 assembly step |
| `optimistix.LevenbergMarquardt` does not converge for ill-conditioned QFM problems | low | medium | allow the caller to select `optimistix.BFGS` or the JAX BFGS penalty path explicitly; do not auto-switch solvers |
| CUDA toolchain mismatch for new kernels | medium | medium | validate on Perlmutter or the current GPU CI lane with the pinned CUDA/JAX runtime |
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

- [x] Do we want to expose a `compute_trapped_fraction_jax` that
  accepts the *frozen* VMEC state directly (skipping
  `(modB, sqrtg)` reconstruction), or keep the current public
  signature only? Recommendation: keep the public signature,
  introduce a private `_jax_core` entrypoint that consumes the
  frozen state. Decision for this milestone: keep the public signature
  as `(modB, sqrtg)`. A frozen-state convenience wrapper would duplicate
  VMEC geometry ownership and is deferred until a measured caller needs
  it.
- [x] For `ProfileSplineJAX`, ship the SciPy FITPACK coefficients at
  the explicit DOF-update boundary and replay them through the JAX
  spline evaluator. Do not re-fit on-device. Resolved by N2 with
  `tests/mhd/test_profiles_jax.py::test_profile_spline_jax_fits_once_per_dof_state`.
- [x] Should the augmented-Lagrangian QFM path expose a *constraint
  Jacobian* hook so users can plug in cheaper closed-form label
  gradients? Recommendation: yes, default to `jax.jacfwd` and accept
  an optional override. Decision for this milestone: do not add the hook
  yet. The current QFM AL path keeps the label derivative inside the
  JAX objective/KKT SSOT; a public override would be a new API surface
  and is deferred until profiling identifies a real closed-form caller.
- [x] Multi-host `shard_map` (across nodes via `jax.distributed`) is
  out of scope for N6 — confirm this is acceptable for the current
  shipping target, or escalate to a separate milestone. Decision:
  acceptable for this shipping target. N6 covers one-node explicit
  `shard_map` with a real four-CUDA-device mesh; multi-host
  `jax.distributed` is a separate milestone if needed.

---

*End of plan.*
