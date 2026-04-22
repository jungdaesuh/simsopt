# CLAUDE.md — simsopt-jax

## What This Is

JAX worktree of simsopt for GPU-accelerated stellarator optimization.
Branch: `jax-port`. Parent repo: `columbia/simsopt`.

## Environment

Two environments are relevant:

**Shared JAX 0.9.2 runtime** (reference `scipy` lane plus private optimizer lane):
```bash
conda env create -f envs/jax-0.9.2.yml
conda activate jax-0.9.2
```
- JAX 0.9.2, jaxlib 0.9.2, NumPy 2.x, Python 3.11
- env recipe provides the build toolchain and performs the editable
  `simsopt[JAX,dev]` install used by local validation, including `ruff`
- use this lane for import smoke, pure-JAX unit tests, Stage 2 parity, and the
  public CPU/GPU parity work

**Private optimizer lane** (`optimizer_backend="hybrid"` / `"ondevice"`):
```bash
conda activate jax-0.9.2
pip install -e .
```
- JAX 0.9.2, jaxlib 0.9.2, NumPy 2.x, Python 3.11
- requires a full simsoptpp-backed editable install
- use this lane for the private optimizer unit/integration tests and real
  `run_code()` validation

**M2 integration tests** (needs simsoptpp for CPU parity):
```bash
/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python -m pytest tests/integration/ -v
```
- Python 3.11, simsoptpp built from `68c6124b`
- `tests/integration/conftest.py` patches the scikit-build meta path finder to inject JAX modules

## Validation

After every code change, run lint, format, and tests:

```bash
ruff check <changed-files>
ruff format <changed-files>

# Public pure-JAX unit tests (no simsoptpp)
conda run -n jax-0.9.2 python -m pytest tests/test_jax_import_smoke.py tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py tests/integration/test_jax_native_path.py -m "not private_optimizer_runtime" -v

# Private optimizer tests (same 0.9.2 runtime, simsoptpp-backed install)
conda run -n jax-0.9.2 python -m pytest tests/geo/test_boozersurface_jax.py tests/integration/test_single_stage_jax.py -m "private_optimizer_runtime" -v

# Benchmark/runtime helper regressions
conda run -n jax-0.9.2 python -m pytest tests/test_run_code_benchmark_common.py tests/test_benchmark_helpers.py -v

# M2+M5 integration tests (needs simsoptpp) — 37 pass
/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python -m pytest tests/integration/ -v
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

## Key Conventions

- **Tensor convention**: `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is derivative direction, axis 2 is B component. Matches SIMSOPT `fields.rst`.
- **No simsoptpp dependency**: Pure JAX modules (M1) use `importlib.util` direct loading in tests to avoid triggering `simsopt/__init__.py` → `simsoptpp`. M2 adapter modules import from `simsopt._core` and are guarded by `try/except ImportError` in `__init__.py`.
- **Parity evidence scope**: direct simsoptpp-backed parity is established for `biot_savart_B`, `surface_gamma`, and `integral_BdotN`. The derivative-heavy pieces (`dB/dX`, surface derivatives, Boozer residual derivatives) still rely on FD / analytical / CPU-oracle checks rather than universal direct C++ parity.
- **Stellsym DOF convention**: `stellsym_scatter_indices(mpol, ntor)` uses cos-cos + sin-sin for x, and cos-sin + sin-cos for y and z (y transforms like z under stellarator symmetry). This matches the CPU `SurfaceXYZTensorFourier` DOF ordering exactly (verified by comparing scatter indices against CPU DOF-to-coefficient probing).
- **Boozer grad/hessian**: M1 wrappers only differentiate through iota/G. Surface DOF derivatives require the composed pipeline (M3+).
- **M3 composed derivatives**: `boozer_penalty_composed()`, `boozer_penalty_grad_composed()`, `boozer_residual_jacobian_composed()`, `boozer_residual_coil_vjp()` in `boozer_residual_jax.py` — pure Boozer pipeline without label constraints.
- **M4 VJP calling convention**: The JAX VJP hooks stored in `res['vjp']` have signature `(lm, booz_surf, iota, G)`, NOT the CPU signature `(lm, booz_surf)`. This is because JAX VJPs construct the decision vector from explicit args rather than reading `booz_surf` internal state.
- **M5 implicit differentiation**: `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` use the IFT adjoint formula: `dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils` where `adj = PLU^{−T} ∂J/∂x_inner`. The PLU and VJP come from `BoozerSurfaceJAX.run_code()` result dict. **Validation status**: strong but split. Fixed-surface FD validates the direct-term `BoozerResidualJAX` path; deterministic adjoint-solve consistency validates the PLU solve; reduced-real re-solve FD validates `IotasJAX` and `NonQuasiSymmetricRatioJAX` on branch-stable samples. End-to-end re-solve FD for the full composed derivative when the adjoint term materially matters is still deferred pending a stable representative fixture.
- **Exact Boozer scaling-limit contract**: exact Newton remains matrix-free inside the loop, but the final dense Jacobian/PLU step is still size-limited. Public exact results now report `failure_category="scaling_limit"`, `failure_stage="dense_jacobian_finalization"`, `jacobian_materialized`, `dense_jacobian_shape`, `dense_jacobian_bytes`, and `max_dense_jacobian_bytes`. Treat that as a predictable exact-mode size ceiling, not generic Newton instability.
- **M5 adapter pattern**: The JAX objective wrappers use CPU surface objects (`surface.gamma()`, `label.J()`) for value computation, and JAX autodiff through `_surface_geometry_from_dofs`/`biot_savart_B` for gradient computation. This is by design (M0 contract adapter pattern): CPU objects at the boundary, JAX on the gradient hot path.
- **Traceable runtime bundle cache contract**: `make_traceable_objective_runtime_bundle()` caches compiled entrypoints against deterministic signatures of the solved baseline state, objective kwargs, and coil/runtime specs. Rebuild the bundle after changing those inputs; do not mutate captured objects and expect an existing cached bundle to retarget itself.
- **Adjoint / warm-start PLU solves**: the wrapper path intentionally uses iterative refinement for adjoint and warm-start solves through stored PLU factors. This is a numerical-stability choice for dense Boozer linearizations, not a fallback to a different physics model.
- **JIT closure strategy**: `SquaredFluxJAX` captures fixed surface arrays (gamma, normal, target) in JIT closures at construction time. Valid for Stage 2 (fixed surface). Do not call `field.set_points()` after constructing `SquaredFluxJAX`.
- **GPU reproducibility policy fields**: `BackendPolicy.gpu_reduction_order_*`, `gpu_reproducibility_*`, and `tolerance_ratchet_factor` are reporting/acceptance metadata for parity lanes. They document tolerance budgets and diagnostic defaults. For CUDA parity lanes, runtime configuration validates that a deterministic XLA GPU flag was set before JAX initialization, but these fields do not directly force kernel execution behavior by themselves.
- **Mixed quadrature support**: `BiotSavartJAX._extract_coil_data_grouped()` groups coils by quadrature point count, evaluates each group via `biot_savart_B`, and sums. This allows TF coils (15-point) and banana coils (128-point) to coexist. The `SquaredFluxJAX` fallback path uses `field.B()` + `jax.grad(integral)` + `field.B_vjp()` chain, which also handles mixed quadrature correctly.
- **C++ ANGLE_RECOMPUTE brace pattern**: In `surfacerzfourier.cpp`, the VJP loops use `if(i % ANGLE_RECOMPUTE == 0)` to periodically recompute trig values. These blocks require explicit `{}` braces — bare `if` only guards the first statement, making costerm unconditional. Always add braces when touching these blocks.
- **JAX scalar boundary conversions**: JAX integer/boolean scalars from `jnp` must be cast to `int()`/`bool()` before storing in result dicts consumed by SciPy or NumPy callers. Pattern: `"iter": int(result.nit), "success": bool(result.success)`.
- **BFGS device residency**: `BoozerSurfaceJAX` least-squares solves expose three backends. `optimizer_backend="scipy"` remains the trusted reference backend. `optimizer_backend="ondevice"` and `optimizer_backend="hybrid"` still depend on private line-search internals in `optimizer_jax.py`, but they now target the same JAX 0.9.2 runtime.

## Code Review History

### Comprehensive jax-port code review (2026-03-18)

Bugs fixed:
- `_ensure_solved` crashed with `TypeError` when `booz_surf.res is None`. Fixed with None guard raising `RuntimeError`. Later hardened to also check `res["success"]` — a failed solve must not trust its PLU/VJP state.
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
- **BiotSavartJAX missing d2B_by_dXdX/A/compute**: backlog items, not needed by current consumers.
- **SciPy host loop in optimizer**: not a bug. It remains the default least-squares backend, but the on-device and hybrid backends are now supported and validated separately.
- **LS solve divergence CPU vs JAX**: did not reproduce — both converge to machine precision.

## Plan

- `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` — full milestone plan
- `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` — M0 contract decisions
