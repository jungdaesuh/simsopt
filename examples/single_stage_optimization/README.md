# Stellarator Banana Coil Optimization Workflow

This repository contains a two-stage optimization workflow for designing banana coils in a stellarator configuration. The workflow optimizes coil geometries to minimize magnetic field errors while satisfying various engineering constraints.

## Overview

The optimization process consists of two sequential stages:

1. **Stage 2 (Coil Design)**: `banana_coil_solver.py` - Optimizes the banana coil geometry to minimize normal magnetic field errors on the plasma surface
2. **Single Stage (Quasi-Symmetry)**: `single_stage_banana_example.py` - Further optimizes the system for quasi-symmetry and Boozer coordinates using outputs from Stage 2

There are also two workflow entrypoints that sit on top of those scripts:

- **Locked 80 kA baseline sweep**: `run_80ka_baseline_tradeoff_sweep.py`
- **Finite-current smoke validation**: `run_finite_current_smoke.py`

These two workflows are intentionally separate:

- The `80 kA` baseline lane is `coil-only, zero-plasma-current`.
- The finite-current lane reuses a frozen Stage 2 artifact and varies only plasma current.
- User-facing plasma current is specified in SI `A` via `--plasma-current-A`.
- Raw `--boozer-I` remains an expert/internal input, not the recommended public interface.

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

- Python 3.x
- SIMSOPT library
- NumPy
- SciPy
- Matplotlib
- Shapely
- Numba
- Bentley_Ottmann (Version 8.0.0)

Install dependencies:
```bash
pip install numpy scipy matplotlib shapely numba bentley_ottmann==8.0.0
```

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
- `--tf-current-A` / `TF_CURRENT_A`: TF current per coil in physical amps
- `--cc-threshold` / `CC_THRESHOLD`: Coil-coil spacing threshold
- `--curvature-threshold` / `CURVATURE_THRESHOLD`: Curvature threshold
- `--maxiter` / `MAXITER`: Maximum optimization iterations
- `--ftol` / `FTOL`: L-BFGS-B function change tolerance (default: `1e-15`, effectively lets `maxiter` control termination)
- `--gtol` / `GTOL`: L-BFGS-B projected gradient tolerance (default: `1e-15`)

**Run**:
```bash
cd STAGE_2
python banana_coil_solver.py \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22
```

**Outputs** (in `STAGE_2/outputs-[plasma_filename]/R0=X-s=Y-.../`):
- `biot_savart_opt.json` - **Required input for Single Stage**
- `curves_opt.vtu` - Optimized coil geometries (VTK format)
- `surf_opt.vtu` - Optimized plasma surface (VTK format)
- `CrossSectionPlot.png` - Diagnostic plot
- `results.json` - Optimization summary

**Note the output directory path** - you'll need it for the next step.

### Step 3: Select the Stage 2 Seed

You no longer need to hand-edit `single_stage_banana_example.py`.

The script can resolve the seed either from the local Stage 2 outputs or from the archive database:

- `--stage2-source database` uses `DATABASE/COIL_OPTIMIZATION/outputs/...`
- `--stage2-source local` uses `STAGE_2/outputs-[plasma]/...`
- `--stage2-bs-path /full/path/to/biot_savart_opt.json` overrides both

### Recommended Workflow Entry Points

For current Columbia finite-current work, prefer these wrappers over hand-running the lower-level scripts:

#### A. Locked `80 kA` Coil-Only Baseline Sweep

This workflow is for the baseline lane only:

- TF current per coil is locked to `80000 A`
- plasma current is locked to `0 A`
- the workflow runs a weighted tradeoff sweep and summarizes the non-dominated set afterward
- the script rejects non-baseline Stage 2 artifacts instead of silently drifting

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_80ka_baseline_tradeoff_sweep.py
```

Use `--stage2-bs-path` only when you want to reuse a frozen Stage 2 artifact. The script validates the loaded artifact metadata against the locked baseline identity before launching the sweep.

#### B. Finite-Current Smoke Validation

This workflow is for quick contract validation of the finite-current surrogate:

- it consumes one frozen Stage 2 artifact
- it varies only plasma current in physical amps
- it validates the actual loaded Stage 2 artifact provenance, not just requested CLI defaults

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_finite_current_smoke.py --currents-A 0,8000,-35200
```

This is a workflow/surrogate validation tool, not a self-consistent finite-current equilibrium run.

### Step 4: Run Single Stage - Quasi-Symmetry Optimization

**Purpose**: Optimize for quasi-symmetry and proper Boozer coordinates using the coils from Stage 2.

**Location**: `SINGLE_STAGE/single_stage_banana_example.py`

**Key Parameters** (CLI flags or environment variables):
- `--mpol` / `MPOL`
- `--ntor` / `NTOR`
- `--vol-target` / `VOL_TARGET`
- `--iota-target` / `IOTA_TARGET`
- `--plasma-current-A` / `PLASMA_CURRENT_A`: User-facing enclosed toroidal plasma current in physical amps
- `--boozer-I` / `BOOZER_I`: Expert/internal surrogate current knob; prefer `--plasma-current-A`
- `--cc-dist` / `CC_DIST`
- `--curvature-threshold` / `CURVATURE_THRESHOLD`
- `--stage2-seed-major-radius` / `STAGE2_SEED_MAJOR_RADIUS`
- `--stage2-seed-toroidal-flux` / `STAGE2_SEED_TOROIDAL_FLUX`
- `--stage2-seed-banana-surf-radius` / `STAGE2_SEED_BANANA_SURF_RADIUS`
- `--stage2-bs-path` / `STAGE2_BS_PATH`
- `--hardware-search-mode` / `HARDWARE_SEARCH_MODE`: Search-time realized-hardware gate policy (`hard`, `warn`, `adaptive`)
- `--hardware-search-soft-iterations` / `HARDWARE_SEARCH_SOFT_ITERATIONS`: Adaptive soft-window budget for early continuation
- `--maxiter` / `MAXITER`

**Run**:
```bash
cd SINGLE_STAGE
python single_stage_banana_example.py \
  --stage2-source database \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --stage2-seed-major-radius 0.915 \
  --stage2-seed-toroidal-flux 0.24 \
  --stage2-seed-banana-surf-radius 0.22 \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --output-root ./outputs/CC_convergence-CC7-iota17-vol10
```

For finite-current surrogate runs, prefer `--plasma-current-A` over raw `--boozer-I`. Do not pass both together.

Branch-local implementation notes for this surrogate tree:

- Single-stage reload treats the Stage 2 artifact as the source of truth for TF-coil partitioning. If the loaded `results.json` records `NUM_TF_COILS`, it must agree with `--num-tf-coils`; otherwise the run aborts instead of silently re-slicing `bs.coils`.
- This branch's Boozer residual contract is the `I`-augmented form `(G + \iota I) B - |B|^2 (x_\varphi + \iota x_\theta)`. Public upstream SIMSOPT docs/source may still show the older vacuum-style `G`-only residual, so treat this repository's local source as the authority when reviewing finite-current surrogate behavior.

Search-time realized hardware handling is now explicit:

- `--hardware-search-mode hard` is the default and rejects hardware-invalid trial candidates during search.
- `--hardware-search-mode warn` keeps the run moving but records the realized hardware violation as warning-only.
- `--hardware-search-mode adaptive` uses the same warning-only handling only during the early soft phase, then reverts to hard rejection.
- Final certification remains hard in all modes. A run that ends hardware-invalid still reports failure even if search-time handling was softened.

For the hard-iota seed A/B test, change only the Stage 2 seed and plasma file:

```bash
python single_stage_banana_example.py \
  --stage2-source database \
  --plasma-surf-filename wout_nfp22ginsburg_000_002084_iota20.nc \
  --stage2-seed-major-radius 0.975 \
  --stage2-seed-toroidal-flux 0.24 \
  --stage2-seed-banana-surf-radius 0.22 \
  --iota-target 0.20 \
  --vol-target 0.10
```

**Outputs** (in `outputs/mpol=X-ntor=Y-<hash>/`, where `<hash>` is a deterministic fingerprint of the run config):
- `biot_savart_opt.json` - Final optimized magnetic field
- `surf_opt.json` - Final optimized surface
- `curves_opt.vtu` - Final coil configurations
- `surf_init.vtu`, `surf_opt.vtu` - Initial and optimized surfaces
- `NormPlotInitial.png`, `NormPlotOptimized.png` - Field error diagnostics
- `CrossSectionInitial.png`, `CrossSectionOptimized.png` - Cross-section plots
- `log.txt` - Detailed optimization log

Relevant result metadata now also records:

- `PLASMA_CURRENT_A`
- `STAGE2_TF_CURRENT_A`
- `STAGE2_TF_CURRENT_SUM_ABS_A`
- `HARDWARE_SEARCH_MODE`
- `HARDWARE_SEARCH_SOFT_ITERATIONS`

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
