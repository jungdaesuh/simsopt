# Stellarator Banana Coil Optimization Workflow

This repository contains a two-stage optimization workflow for designing banana coils in a stellarator configuration. The workflow optimizes coil geometries to minimize magnetic field errors while satisfying various engineering constraints.

## Overview

The optimization process consists of two sequential stages:

1. **Stage 2 (Coil Design)**: `banana_coil_solver.py` - Optimizes the banana coil geometry to minimize normal magnetic field errors on the plasma surface
2. **Single Stage (Quasi-Symmetry)**: `single_stage_banana_example.py` - Further optimizes the system for quasi-symmetry and Boozer coordinates using outputs from Stage 2

## Directory Structure

```
.
├── equilibria/                          # Input VMEC equilibrium files
│   └── wout_nfp22ginsburg_000_014417_iota15.nc
├── STAGE_2/                             # Stage 2 script and outputs
│   ├── banana_coil_solver.py
│   ├── banana-scan.sh
│   └── outputs-[plasma_file]/           # Created by Stage 2
│       └── R0=X-s=Y-..../
│           ├── biot_savart_opt.json     # Required for Single Stage
│           ├── curves_opt.vtu
│           ├── surf_opt.vtu
│           └── results.json
├── SINGLE_STAGE/                        # Single stage script and outputs
│   ├── single_stage_banana_example.py
│   ├── single-scan.sh
│   └── outputs/                         # Created by Single Stage
│       └── mpol=X-ntor=Y/
│           ├── biot_savart_init.json    # Required for Poincaré
│           ├── biot_savart_opt.json
│           ├── surf_init.json           # Required for Poincaré
│           ├── surf_opt.json
│           ├── curves_opt.vtu
│           ├── NormPlot*.png
│           ├── CrossSection*.png
│           ├── log.txt
│           └── PoincarePlot.png         # Created by poincare_surfaces.py
└── POINCARE_PLOTTING/                   # Poincaré plot generation (optional)
    ├── poincare_surfaces.py
    └── poincare-plot.sh
```

## Prerequisites

- Python 3.11 is the recommended release-validation interpreter for the JAX
  lanes.
- CPU/JAX development install from the repository root:

```bash
python -m pip install -e ".[deploy]"
```

- CUDA install from the repository root:

```bash
python -m pip install -e ".[deploy_gpu]"
```

The `deploy` and `deploy_gpu` extras are the repository SSOT for this workflow.
As of 2026-05-19, `deploy_gpu` routes through the repo `JAX_GPU` extra, which
uses the official JAX `jax[cuda12]` CUDA wheel family. Current JAX docs also
publish `jax[cuda13]`; use a different CUDA wheel only as an explicit
environment-lane decision.

Runtime environment recipes:

```bash
# CPU reference oracle.
export SIMSOPT_BACKEND_MODE=native_cpu
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

```bash
# JAX CPU parity.
export SIMSOPT_BACKEND_MODE=jax_cpu_parity
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

```bash
# JAX GPU parity. Requires a CUDA-capable JAX install and NVIDIA GPU node.
export SIMSOPT_BACKEND_MODE=jax_gpu_parity
export SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled
export SIMSOPT_JAX_PLATFORM=cuda
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
```

Official JAX policy that matters for this workflow:

- CUDA pip wheels are installed with extras such as `jax[cuda12]` or
  `jax[cuda13]`.
- `JAX_PLATFORMS` is ordered. Every listed platform must initialize, and the
  first listed platform becomes the default.
- JAX preallocates GPU memory by default; proof jobs set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` explicitly so memory usage is visible
  and reproducible.
- For pip-installed CUDA wheels, do not let `LD_LIBRARY_PATH` override the
  bundled NVIDIA libraries.

## Workflow Instructions

### Step 1: Prepare Input Files

Ensure you have a VMEC equilibrium file (`.nc` format) in the `equilibria/` directory. The default filename is:
```
wout_nfp22ginsburg_000_014417_iota15.nc
```

### Step 2: Run Stage 2 - Banana Coil Design

**Purpose**: Optimize banana coil geometry to minimize magnetic field normal component on the plasma surface.

**Location**: `STAGE_2/banana_coil_solver.py`

**Key Parameters** (CLI flags or environment variables):
- `--major-radius` / `MAJOR_RADIUS`: Major radius target
- `--toroidal-flux` / `TOROIDAL_FLUX`: Normalized toroidal flux surface
- `--banana-surf-radius` / `BANANA_SURF_RADIUS`: Coil surface radius
- `--cc-threshold` / `CC_THRESHOLD`: Coil-coil spacing threshold
- `--curvature-threshold` / `CURVATURE_THRESHOLD`: Curvature threshold
- `--maxiter` / `MAXITER`: Maximum optimization iterations
- `--ftol` / `FTOL`: L-BFGS-B function change tolerance (default: `1e-15`, effectively lets `maxiter` control termination)
- `--gtol` / `GTOL`: L-BFGS-B projected gradient tolerance (default: `1e-15`)

**CPU reference run** from the repository root:
```bash
SIMSOPT_BACKEND_MODE=native_cpu \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend cpu \
  --optimizer-backend scipy \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22 \
  --skip-postprocess
```

**JAX CPU parity run** from the repository root:
```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --optimizer-backend ondevice \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22 \
  --skip-postprocess
```

**Outputs** (in `STAGE_2/outputs-[plasma_filename]/R0=X-s=Y-.../`):
- `biot_savart_opt.json` - **Required input for Single Stage**
- `surf_opt.json` - **Required restart surface for production handoff**
- `results.json` - **Required optimization summary and provenance**
- `curves_opt.vtu` - Optimized coil geometries (VTK format, when post-processing is enabled)
- `surf_opt.vtu` - Optimized plasma surface (VTK format, when post-processing is enabled)
- `CrossSectionPlot.png` - Diagnostic plot

**Note the output directory path** - it is the Stage 2 seed handoff directory.

### Step 3: Select the Stage 2 Seed

You no longer need to hand-edit `single_stage_banana_example.py`.

The normal external handoff path is an explicit Stage 2 artifact path:

```bash
export STAGE2_BS_PATH=/path/to/stage2/run/biot_savart_opt.json
```

A production Stage 2 handoff directory must contain:

- `results.json`
- `biot_savart_opt.json`
- `surf_opt.json`

Rank candidate Stage 2 directories before using them downstream:

```bash
python examples/single_stage_optimization/STAGE_2/stage2_seed_report.py \
  --scan-root examples/single_stage_optimization/STAGE_2 \
  --output-json .artifacts/stage2_seed_catalog.json \
  --require-pass
```

`--stage2-bs-path /path/to/biot_savart_opt.json` overrides all derived seed
resolution. `--stage2-source local` remains useful for scanning local
`STAGE_2/outputs-[plasma]/...` runs. `--stage2-source database` is an
internal/archive option for historical Columbia paths, not the external default.

The checked-in reduced fixture
`benchmarks/fixtures/single_stage_seed_iota15/` is the canonical small
copy-paste fixture for startup/proof commands. It contains `results.json`,
`biot_savart_opt.json`, and `single_stage_jax_runtime_spec.json`; it is not a
complete Stage 2 seed-catalog candidate because it intentionally does not carry
`surf_opt.json`.

### Step 4: Run Single Stage - Quasi-Symmetry Optimization

**Purpose**: Optimize for quasi-symmetry and proper Boozer coordinates using
the coils from Stage 2.

**Location**: `SINGLE_STAGE/single_stage_banana_example.py`

**CPU reference init proof**:

```bash
SIMSOPT_BACKEND_MODE=native_cpu \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend cpu \
  --optimizer-backend scipy \
  --stage2-bs-path benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --init-only \
  --minimal-artifacts \
  --output-root .artifacts/single_stage_cpu_init
```

Production JAX startup uses immutable runtime seed specs on the target lane.
Compile a spec from a warm-start run first:

```bash
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --warm-start-run-dir /path/to/single_stage/warm_start_run \
  --compile-jax-runtime-seed-spec \
  --jax-runtime-seed-spec /path/to/single_stage_jax_runtime_spec.json
```

For the checked-in fixture, use the precompiled spec directly:

```bash
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --init-only \
  --minimal-artifacts \
  --output-root .artifacts/single_stage_jax_cpu_init
```

For GPU parity, keep the same runtime seed spec and change only the execution
lane:

```bash
SIMSOPT_BACKEND_MODE=jax_gpu_parity \
SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled \
SIMSOPT_JAX_PLATFORM=cuda \
JAX_ENABLE_X64=1 \
JAX_PLATFORMS=cuda,cpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env -u LD_LIBRARY_PATH \
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax \
  --optimizer-backend ondevice \
  --boozer-optimizer-backend ondevice \
  --jax-runtime-seed-spec benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --init-only \
  --minimal-artifacts \
  --output-root .artifacts/single_stage_jax_gpu_init
```

**Key Parameters** (CLI flags or environment variables):
- `--mpol` / `MPOL`
- `--ntor` / `NTOR`
- `--vol-target` / `VOL_TARGET`
- `--iota-target` / `IOTA_TARGET`
- `--cc-dist` / `CC_DIST`
- `--curvature-threshold` / `CURVATURE_THRESHOLD`
- `--stage2-seed-major-radius` / `STAGE2_SEED_MAJOR_RADIUS`
- `--stage2-seed-toroidal-flux` / `STAGE2_SEED_TOROIDAL_FLUX`
- `--stage2-seed-banana-surf-radius` / `STAGE2_SEED_BANANA_SURF_RADIUS`
- `--stage2-bs-path` / `STAGE2_BS_PATH`
- `--jax-runtime-seed-spec` / `JAX_RUNTIME_SEED_SPEC`
- `--compile-jax-runtime-seed-spec`
- `--maxiter` / `MAXITER`

**Outputs** (in `outputs/mpol=X-ntor=Y-<hash>/`, where `<hash>` is a deterministic fingerprint of the run config):
- `biot_savart_opt.json` - Final optimized magnetic field
- `surf_opt.json` - Final optimized surface
- `single_stage_jax_runtime_spec.json` - Immutable JAX restart/startup spec
- `curves_opt.vtu` - Final coil configurations
- `surf_init.vtu`, `surf_opt.vtu` - Initial and optimized surfaces
- `NormPlotInitial.png`, `NormPlotOptimized.png` - Field error diagnostics
- `CrossSectionInitial.png`, `CrossSectionOptimized.png` - Cross-section plots
- `log.txt` - Detailed optimization log

### Step 5 (Optional): Generate Poincaré Plots

**Purpose**: Visualize field line topology and magnetic surface quality by generating Poincaré plots.

**Location**: `POINCARE_PLOTTING/poincare_surfaces.py`

**Key Parameters** (editable in script):
- `nfieldlines`: Number of field lines to trace (default: 50)
- `tmax_fl`: Maximum toroidal angle for integration (default: 7000)
- `tol`: Tolerance for field line integration (default: 1e-7)
- `interpolate`: Use interpolated field for faster calculation (default: True)
- `nr`, `nphi`, `nz`: Grid resolution for interpolation (default: 20, 10, 10)
- `degree`: Interpolation degree (default: 3)

**Output directory**: By default, the script auto-selects the most recent
`SINGLE_STAGE/outputs/mpol=*` run directory. Override with `POINCARE_OUT_DIR`:
```bash
export POINCARE_OUT_DIR=/path/to/single_stage/outputs/mpol=8-ntor=6-abcd1234
```

**Run** (the shell script defaults `SIMSOPT_ROOT` to `$HOME/simsopt`; override for non-default checkouts):
```bash
export SIMSOPT_ROOT=/path/to/simsopt   # only if repo is not at ~/simsopt
sbatch POINCARE_PLOTTING/poincare-plot.sh
```

**Outputs** (in the specified `OUT_DIR`):
- `PoincarePlot.png` - Poincaré sections at multiple toroidal angles showing field line intersections

**Interpretation**:
- Well-nested, closed contours indicate good magnetic surfaces
- Scattered or chaotic patterns suggest field line stochasticity
- The black outline shows the plasma surface boundary
- Field lines are traced from the plasma edge outward to check confinement

## Key Optimization Objectives

### Stage 2 (Banana Coil Solver)
- **Squared Flux**: Minimize `B dot n` (normal field on plasma surface)
- **Curve Length**: Penalize coils longer than target (1.75 m)
- **Coil-Coil Distance**: Maintain minimum separation (5 cm)
- **Curvature**: Limit maximum curvature (threshold: 40 m^-1)

### Single Stage
- **Quasi-Symmetry**: Minimize non-quasi-symmetric ratio
- **Boozer Residual**: Minimize Boozer coordinate residual
- **Iota Control**: Maintain target rotational transform
- **Curve Length**: Control coil length
- **Distance Penalties**: Coil-coil, coil-surface, surface-vessel separations
- **Curvature**: Limit coil curvature

## Visualization

Output VTK files can be visualized using:
- **ParaView**: Open `.vtu` files to view 3D geometries
- **VisIt**: Alternative visualization tool

PNG diagnostic plots are generated automatically:
- Cross-section plots show coil and surface geometries
- Normal field plots show magnetic field errors
- Poincaré plots show field line topology and magnetic surface quality

## Troubleshooting

### Stage 2 Issues
- **Self-intersecting coils**: Reduce `CURVATURE_WEIGHT` or adjust coil initialization
- **Optimization not converging**: Increase `MAXITER` or adjust weight parameters
- **High field errors**: Reduce `LENGTH_WEIGHT` or adjust coil surface radius
- **Slow convergence**: Default `--ftol`/`--gtol` of `1e-15` lets `maxiter` control termination; loosen to `--ftol 1e-9 --gtol 1e-5` for faster runs

### Single Stage Issues
- **File not found error**: Verify the path to `biot_savart_opt.json` from Stage 2
- **Boozer surface rejected**: Surface is self-intersecting or solver failed; try different initial conditions
- **Convergence issues**: Adjust `ftol` and `gtol` tolerances for your `mpol` value

### Poincaré Plotting Issues
- **File not found error**: Verify paths to `biot_savart_init.json` and `surf_init.json` from Single Stage
- **Field line integration errors**: Reduce `tol` for higher accuracy or decrease `tmax_fl` if lines escape
- **Memory issues with interpolation**: Set `interpolate=False` or reduce `nr`, `nphi`, `nz` grid resolution
- **Chaotic field lines**: May indicate issues with magnetic configuration; check quasi-symmetry metrics

## Customization

### Changing Plasma Equilibrium
1. Place new VMEC `.nc` file in `equilibria/`
2. Update `plasma_surf_filename` in both scripts
3. Re-run both stages

### Adjusting Optimization Weights
Edit weight parameters in the scripts:
- Stage 2: `LENGTH_WEIGHT`, `CC_WEIGHT`, `CURVATURE_WEIGHT`
- Single Stage: `RES_WEIGHT`, `IOTAS_WEIGHT`, `CC_WEIGHT`, `CS_WEIGHT`, etc.

### Convergence Tolerances

**Stage 2** defaults to `ftol=1e-15` and `gtol=1e-15` (`factr ≈ 4.5`), which effectively
lets `--maxiter` control termination. Override for faster convergence:
```bash
python banana_coil_solver.py --ftol 1e-9 --gtol 1e-5   # scipy "moderate accuracy"
```

**Single stage** includes automatic tolerance adjustment by `mpol`:
```python
ftol_by_mpol = {8: 1e-5, 9: 5e-6, 10: 1e-6, ...}
gtol_by_mpol = {8: 1e-2, 9: 5e-3, 10: 1e-3, ...}
```

### Customizing Poincaré Plots
Adjust field line tracing parameters in `poincare_surfaces.py`:
- Increase `nfieldlines` for denser coverage (impacts computation time)
- Adjust `tmax_fl` to trace field lines for longer/shorter distances
- Set `interpolate=False` for exact field evaluation (slower but more accurate)
- Modify `rrange`, `phirange`, `zrange` to change interpolation grid resolution

## Output Interpretation

### log.txt (Single Stage)
Monitor optimization progress with:
- `Objective J`: Total objective function value
- `||grad J||`: Gradient norm (should decrease)
- `nonQS ratio`: Quasi-symmetry metric
- `Boozer Residual`: Boozer coordinate accuracy
- `Iotas (actual)`: Current rotational transform
- `Volume`: Plasma volume
- `<|B dot n|>`: Average normal field (should be small)

### results.json (Stage 2)
Contains:
- Final parameters and weights used
- Field error metric
- Self-intersection status
- Iteration count

### PoincarePlot.png (Poincaré Analysis)
Visualizes magnetic field line topology:
- **Well-nested closed contours**: Indicate good magnetic surfaces and confinement
- **Scattered/chaotic patterns**: Suggest field line stochasticity and poor confinement
- **Black outline**: Shows the plasma surface boundary
- **Multiple panels**: Different toroidal angle cross-sections (typically 4 per field period)
- Field lines are traced from the plasma edge outward to assess confinement quality 
