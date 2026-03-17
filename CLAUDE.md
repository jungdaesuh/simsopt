# CLAUDE.md — simsopt-jax

## What This Is

JAX worktree of simsopt for GPU-accelerated stellarator optimization.
Branch: `jax-port`. Parent repo: `columbia/simsopt`.

## Environment

Two environments are relevant:

**M1 pure-JAX tests** (no simsoptpp):
```bash
conda run -n columbia-repro-b4815f18 <command>
```
- JAX 0.6.2 (CPU), numpy 1.26.4, Python 3.10
- simsoptpp (C++ extension) is NOT installed in this env
- For GPU: install `jaxlib[cuda12]` and set `JAX_PLATFORMS=cuda`

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

# M1–M4 unit tests (no simsoptpp) — 84 pass, 5 skip
conda run -n columbia-repro-b4815f18 python -m pytest tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py tests/geo/test_boozer_derivatives_jax.py tests/geo/test_boozersurface_jax.py -v

# M2+M5 integration tests (needs simsoptpp) — 19 pass
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
| `tests/integration/test_single_stage_jax.py` | Value parity, FD gradient validation, composite objective, short optimization |

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
- **Stellsym DOF convention**: `stellsym_scatter_indices(mpol, ntor)` uses cos-cos + sin-sin for x, and cos-sin + sin-cos for y and z (y transforms like z under stellarator symmetry). This matches the CPU `SurfaceXYZTensorFourier` DOF ordering exactly (verified by comparing scatter indices against CPU DOF-to-coefficient probing).
- **Boozer grad/hessian**: M1 wrappers only differentiate through iota/G. Surface DOF derivatives require the composed pipeline (M3+).
- **M3 composed derivatives**: `boozer_penalty_composed()`, `boozer_penalty_grad_composed()`, `boozer_residual_jacobian_composed()`, `boozer_residual_coil_vjp()` in `boozer_residual_jax.py` — pure Boozer pipeline without label constraints.
- **M4 VJP calling convention**: The JAX VJP hooks stored in `res['vjp']` have signature `(lm, booz_surf, iota, G)`, NOT the CPU signature `(lm, booz_surf)`. This is because JAX VJPs construct the decision vector from explicit args rather than reading `booz_surf` internal state.
- **M5 implicit differentiation**: `BoozerResidualJAX`, `IotasJAX`, `NonQuasiSymmetricRatioJAX` use the IFT adjoint formula: `dJ/d_coils = ∂J/∂coils − adj^T ∂g/∂coils` where `adj = PLU^{−T} ∂J/∂x_inner`. The PLU and VJP come from `BoozerSurfaceJAX.run_code()` result dict. FD-validated via two complementary tests: (1) fixed-surface FD validates the direct B→coil term (`rel_err < 1e-10`); (2) `IotasJAX.dJ()` re-solve FD validates the full adjoint pipeline (`rel_err < 1e-6`).
- **M5 adapter pattern**: The JAX objective wrappers use CPU surface objects (`surface.gamma()`, `label.J()`) for value computation, and JAX autodiff through `_surface_geometry_from_dofs`/`biot_savart_B` for gradient computation. This is by design (M0 contract adapter pattern): CPU objects at the boundary, JAX on the gradient hot path.
- **JIT closure strategy**: `SquaredFluxJAX` captures fixed surface arrays (gamma, normal, target) in JIT closures at construction time. Valid for Stage 2 (fixed surface). Do not call `field.set_points()` after constructing `SquaredFluxJAX`.
- **Coil data round-trip**: `BiotSavartJAX._extract_coil_data()` reads coil geometry from C++ every call. Acceptable for M2 CPU-mode JIT benefit; GPU-native coil evaluation is a later milestone.

## Plan

- `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` — full milestone plan
- `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` — M0 contract decisions
