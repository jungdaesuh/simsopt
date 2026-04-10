# Stellarator Banana Coil Optimization Workflow

This directory contains the current banana-coil workflow stack for this repository:

- Stage 2 coil optimization in [STAGE_2/banana_coil_solver.py](STAGE_2/banana_coil_solver.py)
- single-stage Boozer / quasi-symmetry optimization in [SINGLE_STAGE/single_stage_banana_example.py](SINGLE_STAGE/single_stage_banana_example.py)
- wrapper workflows in [run_80ka_baseline_tradeoff_sweep.py](run_80ka_baseline_tradeoff_sweep.py) and [run_finite_current_smoke.py](run_finite_current_smoke.py)
- targeted ALM rerun wrapper in [run_nfp10_gil_alm_rerun.py](run_nfp10_gil_alm_rerun.py)
- optional field-line / Poincare diagnostics in [POINCARE_PLOTTING/poincare_surfaces.py](POINCARE_PLOTTING/poincare_surfaces.py)

The codebase has evolved beyond the older "edit script constants and rerun" model. Use CLI flags or environment variables, not source edits, for normal operation.

## Current Workflow Model

The optimization flow is still two-stage:

1. Stage 2 builds a banana-coil artifact rooted at `biot_savart_opt.json` plus a sibling `results.json`.
2. Single-stage consumes that Stage 2 artifact and optimizes Boozer / QS objectives on one or two surfaces.
3. Optional Poincare analysis reads a single-stage output directory and emits validation and diagnostic topology artifacts.

There are two supported wrapper entrypoints for most day-to-day work:

1. `run_80ka_baseline_tradeoff_sweep.py`
   This is the locked coil-only baseline lane.
   It enforces `TF_CURRENT_A = 80000`, `PLASMA_CURRENT_A = 0`, validates any reused Stage 2 artifact against that baseline identity, and sweeps single-stage weights.

2. `run_finite_current_smoke.py`
   This is a finite-current surrogate smoke harness.
   It reuses one frozen coil-only Stage 2 artifact and varies only `--plasma-current-A` through the single-stage surrogate path.

There is also one targeted rerun wrapper for the current NFP10 ALM investigation:

3. `run_nfp10_gil_alm_rerun.py`
   This is a current-branch single-stage ALM lane for the `desc_s024match` NFP10 iota-20 family.
   It pins `--constraint-method alm`, `--alm-formulation gil`, explicit physics thresholds, and warning-mode hardware handling so ALM can see hardware violations instead of hard-rejecting them.

## Branch Scope

This README describes the current behavior of this repository, not necessarily older upstream SIMSOPT examples.

Important branch-local behavior:

- finite current is exposed to users via `--plasma-current-A`
- raw `--boozer-I` still exists, but it is an expert/internal surrogate input
- Stage 2 artifact metadata is validated more strictly than in older versions
- single-stage now supports explicit search-time hardware policies
- single-stage includes optional two-surface mode, topology gating, confinement surrogate scoring, staged Boozer refinement, and basin-hopping controls
- Stage 2 and single-stage both support `--constraint-method {penalty, alm}`

## Directory Layout

```text
examples/single_stage_optimization/
├── README.md
├── equilibria/
├── STAGE_2/
│   ├── banana_coil_solver.py
│   └── outputs-[plasma_filename]/
├── SINGLE_STAGE/
│   ├── single_stage_banana_example.py
│   └── outputs/
├── POINCARE_PLOTTING/
│   ├── poincare_surfaces.py
│   └── poincare-plot.sh
├── banana_opt/
│   └── shared helper modules for Stage 2 / single-stage contracts and objectives
├── workflow_helpers.py
├── workflow_runner_common.py
├── run_80ka_baseline_tradeoff_sweep.py
├── run_finite_current_smoke.py
└── run_nfp10_gil_alm_rerun.py
```

## Prerequisites

You need a working SIMSOPT environment plus the extra Python dependencies used by these examples.

Typical extras:

- `numpy`
- `scipy`
- `matplotlib`
- `shapely`
- `numba`
- `bentley_ottmann==8.0.0`

If you are working inside the repo's normal development environment, prefer that environment over ad hoc installs.

## Recommended Entrypoints

### Locked 80 kA Baseline Sweep

Use this for the current coil-only baseline lane:

- `TF_CURRENT_A = 80000`
- `PLASMA_CURRENT_A = 0`
- frozen baseline Stage 2 contract
- weight sweep over single-stage objectives

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_80ka_baseline_tradeoff_sweep.py
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_80ka_baseline_sweep`
- `--stage2-bs-path` can reuse an existing Stage 2 artifact
- the script validates reused artifacts against the locked baseline contract instead of silently drifting

### Finite-Current Smoke Validation

Use this for quick finite-current surrogate contract checks:

- one frozen Stage 2 artifact
- vary only `--plasma-current-A`
- validate the realized Stage 2 artifact metadata and the single-stage result contract

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_finite_current_smoke.py --currents-A 0,8000,-35200
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_finite_current_smoke`
- this is a surrogate smoke harness, not a self-consistent finite-current equilibrium workflow

### Targeted NFP10 GIL ALM Rerun

Use this when you want the concrete NFP10 single-stage ALM lane discussed in the current audit:

- requires an explicit Stage 2 artifact path
- forces `--alm-formulation gil`
- forces warning-mode hardware handling for ALM because single-surface ALM keeps `gate_scale=1.0`, so adaptive mode would fall back to hard rejection
- writes a compact rerun summary JSON in addition to the normal single-stage artifacts

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_nfp10_gil_alm_rerun.py \
  --stage2-bs-path /full/path/to/biot_savart_opt.json
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_nfp10_gil_alm_rerun`
- all `gil` thresholds and ALM trust-region settings remain CLI-overridable
- `--dry-run` prints and records the exact single-stage command without launching it

## Manual Stage 2

Use [banana_coil_solver.py](STAGE_2/banana_coil_solver.py) when you want to generate or inspect a Stage 2 artifact directly.

Basic penalty-mode example:

```bash
cd /path/to/simsopt-surrogate/examples/single_stage_optimization/STAGE_2
python banana_coil_solver.py \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.915 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.22 \
  --constraint-method penalty
```

Key Stage 2 controls:

- seed / geometry:
  `--plasma-surf-filename`, `--equilibria-dir`, `--major-radius`, `--toroidal-flux`, `--banana-surf-radius`, `--tf-current-A`, `--order`
- optimization weights:
  `--length-weight`, `--cc-weight`, `--cc-threshold`, `--curvature-weight`, `--curvature-threshold`
- optimizer controls:
  `--maxiter`, `--ftol`, `--gtol`
- ALM path:
  `--constraint-method alm`, `--alm-max-outer-iters`, `--alm-penalty-init`, `--alm-penalty-scale`, `--alm-feas-tol`, `--alm-stationarity-tol`, `--alm-trust-radius-*`, `--alm-max-inner-attempts`, `--alm-max-subproblem-continuations`, `--alm-distance-smoothing`, `--alm-curvature-smoothing`, `--alm-taylor-test`
- basin-hopping path:
  `--basin-hops`, `--basin-stepsize`, `--basin-temperature`, `--basin-niter-success`, `--basin-seed`

Operational note:

- `--basin-hops` is only supported in penalty mode
- `--constraint-method=alm` and Stage 2 basin-hopping are mutually exclusive in current code

Stage 2 output root layout:

- `STAGE_2/outputs-[plasma_filename]/...`
- the artifact consumed by single-stage is the generated `biot_savart_opt.json`
- the sibling `results.json` is part of the contract and is now used for stricter validation and provenance

## Stage 2 Seed Resolution For Single-Stage

Single-stage can locate the Stage 2 seed in three ways:

1. Explicit artifact path:
   `--stage2-bs-path /full/path/to/biot_savart_opt.json`

2. Database lookup:
   `--stage2-source database`

3. Local Stage 2 outputs:
   `--stage2-source local`

Additional path controls:

- `--local-stage2-root`
- `--database-stage2-root`

If you do not pass `--stage2-bs-path`, single-stage resolves a Stage 2 seed using the requested seed metadata:

- `--stage2-seed-major-radius`
- `--stage2-seed-toroidal-flux`
- `--stage2-seed-length-weight`
- `--stage2-seed-cc-weight`
- `--stage2-seed-curvature-weight`
- `--stage2-seed-cc-threshold`
- `--stage2-seed-curvature-threshold`
- `--stage2-seed-banana-surf-radius`
- `--stage2-seed-tf-current-A`
- `--stage2-seed-order`

For the common nfp22 example equilibria, defaults are filled automatically when those seed parameters are omitted.

Important caveat:

- those built-in single-stage seed defaults are legacy direct-script defaults, not the locked wrapper baseline
- in particular, the built-in nfp22 seed defaults still include `stage2_seed_tf_current_A = 1.0e5`
- if you need the current locked baseline lane, use [run_80ka_baseline_tradeoff_sweep.py](run_80ka_baseline_tradeoff_sweep.py) or pass the Stage 2 seed / artifact explicitly instead of relying on implicit defaults

## Manual Single-Stage

Use [single_stage_banana_example.py](SINGLE_STAGE/single_stage_banana_example.py) when you want direct control of the single-stage run.

Basic example:

```bash
cd /path/to/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE
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
  --constraint-method penalty
```

ALM operational note:

- single-stage ALM now writes `alm_state.partial.json` inside the run directory at outer-loop transitions and after each recorded ALM history event, so stalled or interrupted runs still leave penalty / multiplier / feasibility diagnostics
- when you are using `--constraint-method alm` in the current single-surface workflow, use `--hardware-search-mode warn` or the dedicated `run_nfp10_gil_alm_rerun.py` wrapper so trial states can expose constraint violations to ALM instead of being hard-rejected immediately

Current high-level flag groups:

- core problem setup:
  `--plasma-surf-filename`, `--equilibria-dir`, `--equilibrium-path`, `--output-root`, `--mpol`, `--ntor`, `--nphi`, `--ntheta`, `--vol-target`, `--iota-target`, `--banana-surf-radius`
- current inputs:
  `--plasma-current-A`, `--boozer-I`, `--num-tf-coils`
- weights / thresholds:
  `--res-weight`, `--iotas-weight`, `--cc-weight`, `--cc-dist`, `--curvature-weight`, `--curvature-threshold`, `--length-weight`, `--cs-weight`, `--cs-dist`, `--surf-dist-weight`, `--ss-dist`
- optimizer controls:
  `--maxiter`, `--maxcor`, `--ftol`, `--gtol`
- Stage 2 seed resolution:
  `--stage2-source`, `--stage2-bs-path`, `--local-stage2-root`, `--database-stage2-root`, plus the `--stage2-seed-*` family
- ALM path:
  `--constraint-method alm`, `--alm-max-outer-iters`, `--alm-penalty-init`, `--alm-penalty-scale`, `--alm-feas-tol`, `--alm-stationarity-tol`, `--alm-trust-radius-*`, `--alm-max-inner-attempts`, `--alm-max-subproblem-continuations`, `--alm-distance-smoothing`, `--alm-curvature-smoothing`
- staged Boozer refinement:
  `--boozer-stage`, `--boozer-stage-refinement`, `--refinement-boozer-stage`, `--refinement-maxiter`, `--refinement-chunk-maxiter`, `--refinement-max-stalled-chunks`
- two-surface mode:
  `--num-surfaces`, `--inner-surface-ratio`, `--surface-gap-threshold`, `--multisurface-ramp-iterations`, `--inner-surface-initial-weight`, `--multisurface-initial-step-scale`, `--multisurface-initial-step-maxiter`
- topology and confinement scoring:
  `--topology-gate-fieldlines`, `--topology-gate-tmax`, `--topology-gate-tol`, `--topology-gate-survival-threshold`, `--topology-gate-penalty-scale`, `--topology-scorer-every`, `--topology-scorer-nfieldlines`, `--topology-scorer-tmax`, `--confinement-objective-weight`, `--confinement-surrogate-*`
- search-time hardware policy:
  `--hardware-search-mode {hard,warn,adaptive}`, `--hardware-search-soft-iterations`
- basin-hopping path:
  `--basin-hops`, `--basin-stepsize`, `--basin-temperature`, `--basin-niter-success`, `--basin-seed`

Important current behavior:

- prefer `--plasma-current-A` over raw `--boozer-I`
- do not pass both current interfaces together unless you are intentionally working at the internal surrogate layer
- `--constraint-method=alm` currently requires `--num-surfaces=1`
- staged Boozer refinement currently requires penalty mode, single-surface mode, and no basin-hopping
- single-stage basin-hopping is only supported in penalty mode

## Hardware Search Policy

Search-time realized hardware handling is explicit in current single-stage code:

- `--hardware-search-mode hard`
  Reject hardware-invalid trial candidates during search.

- `--hardware-search-mode warn`
  Keep the search moving, but record realized hardware violations as warnings during search.

- `--hardware-search-mode adaptive`
  Allow warning-only handling only during the early relaxed search window, then revert to hard rejection.

Final certification is still hard in all modes. A terminal hardware-invalid result is still a failure even if search-time handling was softened.

## Output Artifacts

### Stage 2

Typical Stage 2 outputs include:

- `biot_savart_opt.json`
- `results.json`
- `curves_opt.vtu`
- `surf_opt.vtu`
- cross-section and normal-field diagnostics

Recent code changes also make `results.json` more important than it used to be. It now carries contract fields such as:

- `CONSTRAINT_METHOD`
- Stage 2 geometry / weight metadata
- basin-hopping settings and telemetry when enabled
- ALM metadata when enabled

### Single-Stage

Single-stage writes under `SINGLE_STAGE/outputs/mpol=...-ntor=...-<fingerprint>/`.

Common artifacts:

- `biot_savart_init.json`
- `biot_savart_opt.json`
- `surf_init.json`
- `surf_opt.json`
- `surf_init.vtu`
- `surf_opt.vtu`
- `curves_opt.vtu`
- cross-section and normal-field PNGs
- `log.txt`
- `results.json`

`results.json` now records more workflow contract data than older versions, including items such as:

- `PLASMA_CURRENT_A`
- Stage 2 TF-current provenance fields
- `HARDWARE_SEARCH_MODE`
- `HARDWARE_SEARCH_SOFT_ITERATIONS`
- current-mode provenance
- topology / confinement diagnostics when enabled
- staged Boozer refinement status when enabled

### Poincare

The Poincare script no longer emits only a single `PoincarePlot.png`.

Current outputs include:

- `PoincarePlot_init.png` or `PoincarePlot_opt.png`
- `PoincarePlot_init_diagnostic.png` or `PoincarePlot_opt_diagnostic.png`
- `PoincareMetrics_init.json` or `PoincareMetrics_opt.json`
- `curves_init_poincare*` or `curves_opt_poincare*`
- `surf_init_poincare*` or `surf_opt_poincare*`

By default, the script auto-selects the newest single-stage output under `SINGLE_STAGE/outputs`. Override that with:

```bash
export POINCARE_OUT_DIR=/path/to/SINGLE_STAGE/outputs/mpol=8-ntor=6-...
```

## Poincare Usage

```bash
cd /path/to/simsopt-surrogate/examples/single_stage_optimization
export SIMSOPT_ROOT=/path/to/simsopt
sbatch POINCARE_PLOTTING/poincare-plot.sh
```

Interpretation:

- validation plots stop field lines on Boozer-surface exit
- diagnostic plots use only the box-bounded stopping set
- well-nested closed contours indicate better magnetic surfaces
- scattered or chaotic hits indicate poor confinement / field-line loss

## Troubleshooting

### Stage 2

- self-intersections or poor geometry:
  adjust the geometry weights and thresholds instead of patching source
- slow convergence:
  tune `--maxiter`, `--ftol`, and `--gtol`
- ALM experiments:
  inspect trust-radius and smoothing settings before changing low-level code
- basin-hopping:
  remember that it is penalty-only in current code

### Single-Stage

- ambiguous Stage 2 seed:
  pass `--stage2-bs-path` explicitly
- Stage 2 artifact contract mismatch:
  inspect the Stage 2 sibling `results.json`; single-stage now validates TF partition / provenance more strictly
- near-zero iota collapse in vacuum lanes:
  do not assume this is automatically a zero-current semantic issue; it is often just a bad basin
- tolerance tuning:
  there is no current automatic `mpol`-based tolerance schedule in the script; set `--ftol` and `--gtol` explicitly

### Poincare

- no auto-selected output:
  set `POINCARE_OUT_DIR`
- missing optimized field or surface:
  the script falls back to matching init artifacts when needed
- expensive interpolation:
  reduce interpolation resolution or disable interpolation inside the script if you are intentionally debugging that path

## Customization Guidance

Use CLI flags or environment variables for normal changes.

Do this:

- change equilibrium via `--plasma-surf-filename`, `--equilibria-dir`, or `--equilibrium-path`
- change Stage 2 seed resolution via `--stage2-source`, `--stage2-bs-path`, and `--stage2-seed-*`
- change optimization behavior via the documented flags

Do not rely on this older workflow:

- editing `plasma_surf_filename` directly in source
- editing hard-coded weights in scripts as the normal way to run
- assuming older README-era artifact names are still the full contract

## Related Files

- [workflow_runner_common.py](workflow_runner_common.py)
  Shared wrapper helpers and Stage 2 artifact config / command construction.

- [workflow_helpers.py](workflow_helpers.py)
  Naming and path helpers for Stage 2 and workflow artifact layout.

- [alm_utils.py](alm_utils.py)
  Shared augmented-Lagrangian utilities used by Stage 2 and single-stage.

- [topology_scorer.py](topology_scorer.py)
  Shared topology scoring logic used by single-stage callbacks and Poincare validation.
