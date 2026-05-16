# CLAUDE.md — simsopt-jax

## What This Is

JAX worktree of simsopt for GPU-accelerated stellarator optimization.
Current checkout branch: `gpu-purity-stage2-20260405`. Parent repo:
`columbia/simsopt`.

## Environment

Two environments are relevant:

**Shared JAX runtime** (env name `jax`; current checked local env imports
`jax==0.10.0`, `jaxlib==0.10.0`):
```bash
conda env create -f envs/jax.yml
conda activate jax
```
- Current checked local env: JAX 0.10.0, jaxlib 0.10.0, NumPy 2.x, Python
  3.11. Fresh env resolution follows `pyproject.toml`.
- env recipe provides the build toolchain and performs the editable
  `simsopt[JAX,dev]` install used by local validation, including `ruff`
- use this lane for import smoke, pure-JAX unit tests, Stage 2 parity, and the
  public CPU/GPU parity work
- before recording version-sensitive evidence, verify the imported versions with
  `python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"`

**Private optimizer lane** (`optimizer_backend="ondevice"`):
```bash
conda activate jax
pip install -e .
```
- same checked local JAX 0.10.0 / jaxlib 0.10.0 runtime, NumPy 2.x, Python
  3.11
- requires a full simsoptpp-backed editable install
- use this lane for the private optimizer unit/integration tests and real
  `run_code()` validation

**M2 integration tests** (needs simsoptpp for CPU parity):
```bash
.conda/jax/bin/python -m pytest tests/integration/ -v
```
- the in-tree `.conda/jax` env (created by `envs/jax.yml`) ships
  `simsoptpp` under `site-packages/`, so the integration suite runs without
  reaching outside the worktree
- `tests/integration/conftest.py` patches the scikit-build meta path finder to inject JAX modules

## Validation

After every code change, run lint, format, and tests:

```bash
ruff check <changed-files>
ruff format <changed-files>

# Public pure-JAX unit tests (no simsoptpp)
conda run -n jax python -m pytest tests/test_jax_import_smoke.py tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py tests/integration/test_jax_native_path.py -m "not private_optimizer_runtime" -v

# Private optimizer tests (same checked local 0.10.0 runtime, simsoptpp-backed install)
conda run -n jax python -m pytest tests/geo/test_boozersurface_jax.py tests/integration/test_single_stage_jax.py -m "private_optimizer_runtime" -v

# Benchmark/runtime helper regressions
conda run -n jax python -m pytest tests/test_run_code_benchmark_common.py tests/test_benchmark_helpers.py -v

# M2+M5 integration tests (needs simsoptpp) — 37 pass
.conda/jax/bin/python -m pytest tests/integration/ -v
```

Pre-existing mypy errors from upstream (pybind11 stubs, wildcard imports) are expected. Only zero-regression on files you touched.

## JAX Module Layout

JAX modules live alongside C++ counterparts. They do NOT import simsoptpp.

### M1 — Pure JAX functions (no Optimizable integration)

| Module | Purpose |
|--------|---------|
| `src/simsopt/field/biotsavart_jax.py` | Biot-Savart B + dB/dX (autodiff) |
| `src/simsopt/geo/surface_fourier_jax.py` | SurfaceXYZTensorFourier eval |
| `src/simsopt/geo/boozer_residual_jax.py` | Boozer residual scalar + grad/hessian |
| `src/simsopt/objectives/integral_bdotn_jax.py` | integral_BdotN (3 definitions) |
| `benchmarks/jax_feasibility_spike.py` | Timing harness |

### M2 — Optimizable adapters (Stage 2 JAX field path)

| Module | Purpose |
|--------|---------|
| `src/simsopt/field/biotsavart_jax_backend.py` | `BiotSavartJAX(Optimizable)` — wraps coils, B/dB/VJP via JAX |
| `src/simsopt/objectives/fluxobjective_jax.py` | `SquaredFluxJAX(Optimizable)` — end-to-end JAX autodiff |
| `tests/integration/test_stage2_jax.py` | Parity tests: value, gradient, composite, short run |
| `tests/integration/conftest.py` | Meta path finder patch for cross-env testing |

### M3 — Composed derivative path (Boozer residual derivatives via autodiff)

| Module | Purpose |
|--------|---------|
| `src/simsopt/geo/boozer_residual_jax.py` | M3 additions: `boozer_penalty_composed`, `boozer_penalty_grad_composed`, `boozer_residual_jacobian_composed`, `boozer_residual_coil_vjp` |
| `src/simsopt/geo/surface_fourier_jax.py` | M3 additions: `dgamma_by_dcoeff`, `dgammadash1_by_dcoeff`, `dgammadash2_by_dcoeff` via `jax.jacfwd` |
| `tests/geo/test_boozer_derivatives_jax.py` | 19 FD-validated tests |
| `benchmarks/jax_derivative_benchmark.py` | Timing harness: compile + steady-state |

### M4 — JAX Boozer Solver (inner solve on-device)

| Module | Purpose |
|--------|---------|
| `src/simsopt/geo/boozersurface_jax.py` | `BoozerSurfaceJAX(Optimizable)` — LS + exact solver, VJP hooks |
| `src/simsopt/geo/optimizer_jax.py` | `jax_minimize` (BFGS/L-BFGS adapter), `newton_polish`, `newton_exact` |
| `src/simsopt/geo/label_constraints_jax.py` | `volume_jax`, `area_jax`, `toroidal_flux_jax`, `compute_G_from_currents` |
| `tests/geo/test_boozersurface_jax.py` | 29+ tests: pure functions + adapter class + VJP + exact path |

### M5 — Single-Stage Objective Wrappers (implicit differentiation)

| Module | Purpose |
|--------|---------|
| `src/simsopt/geo/surfaceobjectives_jax.py` | `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` — Optimizable wrappers with IFT gradient |
| `tests/integration/test_single_stage_jax.py` | Value sanity, adjoint consistency, FD gradient, composite pipeline, backend construction, LS parity, short outer opt, exact path, ensure-solved guard (14 tests) |

### M6 — Productionization

| Module | Purpose |
|--------|---------|
| `src/simsopt/backend.py` | Unified backend selection API: `get_backend()`, `is_jax_backend()`, `get_jax_platform()` |
| `.github/workflows/jax_smoke.yml` | CI smoke tests for JAX modules (no simsoptpp needed) |
| `docs/source/jax_gpu_setup.rst` | GPU environment setup + runbook |
| `docs/source/jax_acceptance.rst` | CPU-vs-JAX acceptance criteria for research use |

### Backend selection

Two orthogonal env vars:

```bash
# Code-path backend: cpu (simsoptpp) vs jax
SIMSOPT_BACKEND=jax   # or legacy: STAGE2_BACKEND=jax

# JAX device platform: cpu vs cuda (only when backend=jax)
SIMSOPT_JAX_PLATFORM=cuda   # or legacy: SIMSOPT_JAX_BACKEND=cuda
```

Script usage:
```bash
# Stage 2 on GPU
SIMSOPT_BACKEND=jax SIMSOPT_JAX_PLATFORM=cuda python banana_coil_solver.py

# Single-stage on JAX
python single_stage_banana_example.py --backend jax

# or via env var
SIMSOPT_BACKEND=jax python single_stage_banana_example.py
```

Programmatic:
```python
from simsopt.backend import get_backend, is_jax_backend, get_jax_platform
```

## Parity modes

The strict CPU/JAX byte-identity gate (`_pre_newton_census_gate_failures`
in `benchmarks/single_stage_init_parity.py`) remains a release blocker
for production CI. The runtime supports two orthogonal mode families:

- **`*_parity` modes** (`jax_cpu_parity`, `jax_gpu_parity`) — verification
  lanes. Run cpu_ordered twins, target byte identity to the C++ oracle
  within build, slower (5-20×).
- **`*_fast` modes** (`jax_cpu_fast`, `jax_gpu_fast`) — researcher
  speed-opt-out lanes. Run matmul/jacfwd/einsum hot paths. **Explicitly
  fail the byte-identity gate by construction.** Use for non-publication
  exploration only; pre-publication artifacts MUST come from `*_parity`
  lanes.
- **`native_cpu`** — C++ reference oracle. No JAX involved.

All modes share **one strict gate contract**. The fast lanes do not get
a relaxed tolerance lane; they are speed promises without a parity
claim.

See `docs/parity_dual_mode_contract_2026-05-08.md` for the full
specification (mode matrix, reporting context, DM-A/B/D/E slices).

## Key Conventions

- **Tensor convention**: `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is derivative direction, axis 2 is B component. Matches SIMSOPT `fields.rst`.
- **No simsoptpp dependency**: Pure JAX modules (M1) use `importlib.util` direct loading in tests to avoid triggering `simsopt/__init__.py` → `simsoptpp`. M2 adapter modules import from `simsopt._core` and are guarded by `try/except ImportError` in `__init__.py`.
- **Parity ladder SSOT**: `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` owns the lane-specific precision contract. `rtol=1e-10` only applies to same-state `direct-kernel` C++ parity (`biot_savart_B`, `surface_gamma`, `integral_BdotN`, raw Boozer residual) and the existing same-state `ls-wrapper-gradient` fixture. Derivative-heavy paths (`dB/dX`, surface derivatives, Boozer residual derivatives) are tracked by the `derivative-heavy` lane: direct C++ oracle coverage exists for representative first-derivative kernels (`dB/dX`, Biot-Savart VJP, surface coefficient Jacobians, composed Boozer residual Jacobian) at `rtol=1e-8, atol=1e-10`. Column-complete CPU/C++ Hessian basis-sweep parity is covered by `TestUpstreamFactoryBoozerMatrix::test_penalty_hessian_column_complete_cpu_parity_matrix` in the `direct-hessian-oracle` lane at `rtol=1e-8, atol=1e-10`; the seeded directional HVP test is retained as operator-path coverage.
- **Stellsym DOF convention**: `stellsym_scatter_indices(mpol, ntor)` uses cos-cos + sin-sin for x, and cos-sin + sin-cos for y and z (y transforms like z under stellarator symmetry). This matches the CPU `SurfaceXYZTensorFourier` DOF ordering exactly (verified by comparing scatter indices against CPU DOF-to-coefficient probing).
- **Boozer grad/hessian**: M1 wrappers only differentiate through iota/G. Surface DOF derivatives require the composed pipeline (M3+).
- **M3 composed derivatives**: `boozer_penalty_composed()`, `boozer_penalty_grad_composed()`, `boozer_residual_jacobian_composed()`, `boozer_residual_coil_vjp()` in `boozer_residual_jax.py` — pure Boozer pipeline without label constraints.
- **M4 VJP calling convention**: The JAX VJP hooks stored in `res['vjp']` have signature `(lm, booz_surf, iota, G)`, NOT the CPU signature `(lm, booz_surf)`. This is because JAX VJPs construct the decision vector from explicit args rather than reading `booz_surf` internal state.
- **M5 implicit differentiation**: `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` use the IFT adjoint formula: `dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils` where `adj` solves the transposed inner linearization. `BoozerSurfaceJAX.get_adjoint_runtime_state()` is the source of truth and exposes operator-backed solve callbacks plus streaming grouped VJPs; dense JAX `PLU` entries in result dicts are metadata only. **Validation status**: strong but split by parity-ladder lane. Fixed-surface FD validates the direct-term `BoozerResidualJAX` path (`fd-gradient`); exact operator-status checks validate residual-based success/failure without dense PLU fallback; reduced-real re-solve FD validates `IotasJAX`, `NonQuasiSymmetricRatioJAX`, and a composite BoozerResidual+iota-penalty objective on branch-stable samples (`branch-stable-resolve` / `fd-gradient`). Well-conditioned exact fixtures now exercise operator-vs-dense/PLU adjoint vector parity and IotasJAX gradient projection under `exact-well-conditioned-adjoint` (`rtol=1e-6, atol=1e-8`, residual `<=1e-10`). True ill-conditioned exact fixtures, when present, remain residual/failure-only under `exact-ill-conditioned-adjoint` and must not assert vector parity.
- **Exact Boozer scaling-limit contract**: the `production_operator` exact lane never falls back to dense factors at runtime. The `cpp_compatible_probe` harness materializes a dense host-resident reference exact solver for diagnostic comparison only; it is not exposed through the `BoozerSurfaceJAX` user API and the exact normalizer in `_normalize_solver_options` (`boozersurface_jax.py:3122`, with the strip itself at `boozersurface_jax.py:3185-3186`) continues to drop `optimizer_backend` from the user-visible exact path. Batched exact adjoints in `production_operator` solve one RHS at a time through the same operator seam. Public exact results report `linear_solve_backend="operator"`, `dense_linear_solve_factors_available`, `failure_category="scaling_limit"`, `failure_stage="dense_jacobian_finalization"`, `jacobian_materialized`, `dense_jacobian_shape`, `dense_jacobian_bytes`, and `max_dense_jacobian_bytes`. Treat dense-finalization ceilings as predictable exact-mode reporting limits, not adjoint availability failures.
- **M5 adapter pattern**: The JAX objective wrappers use CPU surface objects (`surface.gamma()`, `label.J()`) for value computation, and JAX autodiff through `_surface_geometry_from_dofs`/`biot_savart_B` for gradient computation. This is by design (M0 contract adapter pattern): CPU objects at the boundary, JAX on the gradient hot path.
- **Traceable runtime bundle cache contract**: `make_traceable_objective_runtime_bundle()` caches compiled entrypoints against deterministic signatures of the solved baseline state, objective kwargs, coil runtime state, coil reconstruction layout, and success-filter contract. The cache key must not use `id()` or per-instance adapter tokens; solved baseline freshness is represented by `BoozerSurfaceJAX._traceable_solve_state_token`, coil DOF freshness by `BiotSavartJAX` / `SpecBackedBiotSavartJAX._coil_dof_state_token`, and coil layout by a structural signature of `coil_dof_extraction_spec`. `BiotSavartJAX` advances its coil-state token both on aggregate `x` / `full_x` writes and on SIMSOPT ancestor DOF invalidation. Success filters should expose `_traceable_runtime_cache_signature` for semantic sharing; otherwise the key holds a live callable-reference signature that compares with `is`, not `id(callable)` or user-defined callable equality. Rebuild the bundle after changing the captured inputs; do not mutate captured objects and expect an existing cached bundle to retarget itself.
- **Adjoint / warm-start operator solves**: JAX wrapper adjoints and traceable warm-start predictors use operator-backed linear solves by default, with the following exception: when `decision_size² × 8 ≤ max_dense_jacobian_bytes`, the LS forward and adjoint solves consume the same `(lu, piv)` factors stored under `lax.stop_gradient` to ensure bit-equal forward/adjoint Hessian action. The LS `(P, L, U)` field is load-bearing runtime data (see `boozersurface_jax.py:3514-3540`, `surfaceobjectives_jax.py:3167-3220`); the **exact** lane's `(P, L, U)` remains debug metadata only, and `BoozerSurfaceJAX.get_adjoint_runtime_state()` remains the runtime SSOT for the exact-lane adjoint. A successful traceable forward solve with a failed adjoint solve must surface a non-finite gradient, not a finite direct-gradient or failure-penalty fallback.
- **Note on `linear_solve_factors`**: the "Adjoint / warm-start operator solves" rule that "dense PLU data in exact results is public/debug metadata" applies to the **exact** lane. In the **LS** lane, the SciPy reference runtime callbacks at `boozersurface_jax.py:3514-3540` build `H_host = P @ L @ U` from `self.res["PLU"]` and use it as `apply_forward`/`apply_transpose`, and the traceable adjoint `_traceable_solve_plu_linearization` at `surfaceobjectives_jax.py:3167-3220` consumes the PLU factors for triangular solves. In those LS paths, `linear_solve_factors` is load-bearing runtime data, not metadata.
- **JIT closure strategy**: `SquaredFluxJAX` captures fixed surface arrays (gamma, normal, target) in JIT closures at construction time. Valid for Stage 2 (fixed surface). Do not call `field.set_points()` after constructing `SquaredFluxJAX`.
- **GPU reproducibility policy fields**: `BackendPolicy.gpu_reduction_order_*`, `gpu_reproducibility_*`, and `tolerance_ratchet_factor` are reporting/acceptance metadata for parity lanes. They document tolerance budgets and diagnostic defaults. For CUDA parity lanes, runtime configuration validates that a deterministic XLA GPU flag was set before JAX initialization, but these fields do not directly force kernel execution behavior by themselves.
- **Mixed quadrature support**: `BiotSavartJAX._extract_coil_data_grouped()` groups coils by quadrature point count, evaluates each group via `biot_savart_B`, and sums. This allows TF coils (15-point) and banana coils (128-point) to coexist. `SquaredFluxJAX` consumes immutable field/surface specs on the native JAX lane; unsupported fields are rejected instead of routing through `field.B()` / `field.B_vjp()` compatibility calls.
- **C++ ANGLE_RECOMPUTE brace pattern**: In `surfacerzfourier.cpp`, the VJP loops use `if(i % ANGLE_RECOMPUTE == 0)` to periodically recompute trig values. These blocks require explicit `{}` braces — bare `if` only guards the first statement, making costerm unconditional. Always add braces when touching these blocks.
- **JAX scalar boundary conversions**: JAX integer/boolean scalars from `jnp` must be cast to `int()`/`bool()` before storing in result dicts consumed by SciPy or NumPy callers. Pattern: `"iter": int(result.nit), "success": bool(result.success)`.
- **BFGS device residency**: `BoozerSurfaceJAX` least-squares solves expose two backends. `optimizer_backend="scipy"` remains the trusted reference backend. `optimizer_backend="ondevice"` still depends on private line-search internals in `optimizer_jax.py`, but it targets the same checked local JAX 0.10.0 runtime. The set of valid backends is enforced by `optimizer_jax.VALID_OPTIMIZER_BACKENDS = {"scipy", "ondevice"}`.
- **Test oracle lint**: New `test_*_jax_*.py` files must cite an independent oracle for every parity assertion (C++ symbol, closed-form expression, FD, or pinned dataset). Re-export `is`-identity, JAX-vs-JAX, and "host wrapper that routes through JAX" comparisons are tautologies. See `tests/REVIEWER_ORACLE_LINT.md`.
- **Floating-point reproducibility across machines**: byte-identity CPU↔JAX state parity is **not** a portable invariant on the LS path. The same source tree on the oversampled BoozerSurface LS parity fixture (`build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)`) produced `sdofs_inf` ranging from `1.9e-14` to `3.6e-12` across two macOS hardware platforms; the Hessian condition number was hardware-invariant (`κ ≈ 5.3e+04`) but the entrywise Hessian disagreement scaled with hardware (`H_inf_diff` from `1.4e-12` to `9.6e-10`). When introducing new state-parity gates, set absolute thresholds at ≥3× the worst measured cross-machine value (currently `sdofs_inf ≤ 1e-11`); reserve `rtol=1e-12` for the same-state direct-kernel lane on a single machine.

## Code Review History

### Comprehensive jax-port code review (2026-03-18)

Bugs fixed:
- `_ensure_solved` crashed with `TypeError` when `booz_surf.res is None`. Fixed with None guard raising `RuntimeError`. Later hardened to also check `res["success"]` — a failed solve must not trust its runtime adjoint state.
- Missing `weight_inv_modB` in exact-path result dict (`boozersurface_jax.py`). Consumers defaulted to wrong value.
- Unconditional `import jax` in `__init__.py` and `surfaceobjectives.py` broke CPU-only installs. Guarded with `try/except ImportError`.
- Missing `}` closing `#pragma omp parallel` in `surfacerzfourier.cpp` `dgamma_by_dcoeff_vjp`.
- `#pragma omp parallel for ordered` without ordered blocks in 3 C++ files. Removed `ordered` clause.
- `mod_B_squared` data race in `integral_BdotN.cpp`. Moved declaration inside loop body.
- Missing braces in ANGLE_RECOMPUTE if-blocks in 3 VJP functions (`surfacerzfourier.cpp`).
- Docstring `r"""` → `"""` regression in `surfaceobjectives.py` (ruff format stripped raw prefix).
- Added `int()`/`bool()` boundary conversions for JAX scalars in `boozersurface_jax.py`.

Confirmed NOT bugs (false positives):
- **nfp factor in volume/area**: correct — nfp cancels with quadrature step `1/(nfp*nphi)`.
- **J_z omission from adjoint source**: correct — inner-solve regularizer, not part of `J_outer`.
- **framedcurve.py API change** (4-arg → 3-arg): all callers already updated.
- **BiotSavartJAX missing `compute(derivatives=N)` bundled cache entrypoint**: classified NON-PORTABLE-by-design (see `.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md`); per-method JAX calls are the canonical JAX-native shape.
- **SciPy host loop in optimizer**: not a bug. It remains the default least-squares backend, but the on-device backend is now supported and validated separately.
- **LS solve divergence CPU vs JAX**: did not reproduce — both converge to machine precision.

## Plan

- `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` — full milestone plan
- `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` — M0 contract decisions
