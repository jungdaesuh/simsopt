# CLAUDE.md — simsopt-jax

## What This Is

JAX worktree of simsopt for GPU-accelerated stellarator optimization.
Branch: `jax-port`. Parent repo: `columbia/simsopt`.

## Environment

```bash
conda run -n columbia-repro-b4815f18 <command>
```

- JAX 0.6.2 (CPU), numpy 1.26.4, Python 3.10
- simsoptpp (C++ extension) is NOT installed in this env
- For GPU: install `jaxlib[cuda12]` and set `JAX_PLATFORMS=cuda`

## Validation

After every code change, run lint, format, and tests:

```bash
conda run -n columbia-repro-b4815f18 ruff check <changed-files>
conda run -n columbia-repro-b4815f18 ruff format <changed-files>
conda run -n columbia-repro-b4815f18 python -m pytest tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py tests/geo/test_boozer_residual_jax.py tests/objectives/test_integral_bdotn_jax.py -v
```

Pre-existing mypy errors from upstream (pybind11 stubs, wildcard imports) are expected. Only zero-regression on files you touched.

## JAX Module Layout

JAX modules live alongside C++ counterparts. They do NOT import simsoptpp.

| Module | Purpose |
|--------|---------|
| `src/simsopt/field/biotsavart_jax.py` | Biot-Savart B + dB/dX (autodiff) |
| `src/simsopt/geo/surface_fourier_jax.py` | SurfaceXYZTensorFourier eval |
| `src/simsopt/geo/boozer_residual_jax.py` | Boozer residual scalar + grad/hessian |
| `src/simsopt/objectives/integral_bdotn_jax.py` | integral_BdotN (3 definitions) |
| `benchmarks/jax_feasibility_spike.py` | Timing harness |

## Key Conventions

- **Tensor convention**: `dB_by_dX[p, j, l] = ∂_j B_l(x_p)` — axis 1 is derivative direction, axis 2 is B component. Matches SIMSOPT `fields.rst`.
- **No simsoptpp dependency**: JAX modules use `importlib.util` direct loading in tests to avoid triggering `simsopt/__init__.py` → `simsoptpp`.
- **Stellsym**: Not yet supported in `surface_gamma_from_dofs`. Use `surface_gamma` with pre-masked coefficient matrices for stellsym surfaces.
- **Boozer grad/hessian**: M1 wrappers only differentiate through iota/G. Surface DOF derivatives require the composed pipeline (M2).

## Plan

- `/Users/suhjungdae/code/columbia/analysis/jax_port_plan.md` — full milestone plan
- `/Users/suhjungdae/code/columbia/analysis/jax_port_m0_contract.md` — M0 contract decisions
