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

**Key Parameters** (editable in script):
- `R0`: Major radius target (default: 0.925 m)
- `s`: Normalized toroidal flux surface (default: 0.24)
- `banana_surf_radius`: Coil surface radius (default: 0.215 m)
- `order`: Fourier modes for coils (default: 2)
- `MAXITER`: Maximum optimization iterations (default: 300)

**Run**:
```bash
cd STAGE_2
sbatch banana-scan.sh
```

**Outputs** (in `STAGE_2/outputs-[plasma_filename]/R0=X-s=Y-.../`):
- `biot_savart_opt.json` - **Required input for Single Stage**
- `curves_opt.vtu` - Optimized coil geometries (VTK format)
- `surf_opt.vtu` - Optimized plasma surface (VTK format)
- `CrossSectionPlot.png` - Diagnostic plot
- `results.json` - Optimization summary

**Note the output directory path** - you'll need it for the next step.

### Step 3: Configure Single Stage Script

**Edit** `single_stage_banana_example.py`:

Update the path to the Stage 2 output directory (around line 42):
```python
bs = load(f'../STAGE_2/outputs-{plasma_surf_filename}/R0=0.925-s=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.215-Order=2/biot_savart_opt.json')
```

Ensure this path matches your Stage 2 output directory.

### Step 4: Run Single Stage - Quasi-Symmetry Optimization

**Purpose**: Optimize for quasi-symmetry and proper Boozer coordinates using the coils from Stage 2.

**Location**: `SINGLE_STAGE/single_stage_banana_example.py`

**Key Parameters** (editable in script):
- `mpol`: Poloidal Fourier modes (default: 8)
- `ntor`: Toroidal Fourier modes (default: 6)
- `vol_target`: Target volume (default: 0.10)
- `iota_target`: Target rotational transform (default: 0.15)
- `MAXITER`: Maximum iterations (default: 300)

**Run**:
```bash
cd SINGLE_STAGE
sbatch single-scan.sh
```

**Outputs** (in `outputs/mpol=X-ntor=Y/`):
- `biot_savart_opt.json` - Final optimized magnetic field
- `surf_opt.json` - Final optimized surface
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

**Edit the output directory path** (around line 16):
```python
OUT_DIR = f'../SINGLE_STAGE/outputs/mpol=8-ntor=6'
```

Ensure this path matches your Single Stage output directory.

**Run**:
```bash
cd POINCARE_PLOTTING
sbatch poincare-plot.sh
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

### Multi-Resolution Strategy
Single stage includes automatic tolerance adjustment by `mpol`:
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
