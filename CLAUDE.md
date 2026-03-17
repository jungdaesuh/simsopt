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

# M1 unit tests (no simsoptpp)
conda run -n columbia-repro-b4815f18 python -m pytest tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py -v

# M2 integration tests (needs simsoptpp)
/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python -m pytest tests/integration/test_stage2_jax.py -v
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

### Backend selection (Stage 2 example)

```bash
# CPU (default)
python banana_coil_solver.py

# JAX
python banana_coil_solver.py --backend jax

# or via env var
STAGE2_BACKEND=jax python banana_coil_solver.py
```

## Key Conventions

- **Tensor convention**: `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is derivative direction, axis 2 is B component. Matches SIMSOPT `fields.rst`.
- **No simsoptpp dependency**: Pure JAX modules (M1) use `importlib.util` direct loading in tests to avoid triggering `simsopt/__init__.py` → `simsoptpp`. M2 adapter modules import from `simsopt._core` and are guarded by `try/except ImportError` in `__init__.py`.
- **Stellsym**: Not yet supported in `surface_gamma_from_dofs` (raises `NotImplementedError`). Use `surface_gamma` with pre-masked coefficient matrices for stellsym surfaces.
- **Boozer grad/hessian**: M1 wrappers only differentiate through iota/G. Surface DOF derivatives require the composed pipeline (M3+).
- **JIT closure strategy**: `SquaredFluxJAX` captures fixed surface arrays (gamma, normal, target) in JIT closures at construction time. Valid for Stage 2 (fixed surface). Do not call `field.set_points()` after constructing `SquaredFluxJAX`.
- **Coil data round-trip**: `BiotSavartJAX._extract_coil_data()` reads coil geometry from C++ every call. Acceptable for M2 CPU-mode JIT benefit; GPU-native coil evaluation is a later milestone.

## Plan

- `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` — full milestone plan
- `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` — M0 contract decisions
