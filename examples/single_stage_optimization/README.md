# Stellarator Banana Coil Optimization Workflow

This directory contains the current banana-coil workflow stack for this repository:

- Stage 2 coil optimization in [STAGE_2/banana_coil_solver.py](STAGE_2/banana_coil_solver.py)
- single-stage Boozer / quasi-symmetry optimization in [SINGLE_STAGE/single_stage_banana_example.py](SINGLE_STAGE/single_stage_banana_example.py)
- wrapper workflows in [run_80ka_baseline_tradeoff_sweep.py](run_80ka_baseline_tradeoff_sweep.py) and [run_finite_current_smoke.py](run_finite_current_smoke.py)
- generic Stage 2 ALM wrapper in [run_stage2_alm.py](run_stage2_alm.py)
- unified Stage 2.5 handoff runner in [run_stage2_to_single_stage.py](run_stage2_to_single_stage.py)
- standalone donor-repair wrapper in [run_single_stage_donor_repair.py](run_single_stage_donor_repair.py)
- Stage 2 iota decision-gate benchmark wrapper in [run_stage2_iota_decision_gate.py](run_stage2_iota_decision_gate.py)
- generic single-stage ALM rerun wrapper in [run_single_stage_thresholded_physics_alm.py](run_single_stage_thresholded_physics_alm.py)
- explicit target-vs-frontier comparison wrapper in [run_single_stage_goal_mode_comparison.py](run_single_stage_goal_mode_comparison.py)
- explicit iota-target sweep wrapper in [run_single_stage_iota_target_sweep.py](run_single_stage_iota_target_sweep.py)
- banana-current scan wrapper in [run_banana_current_scan.py](run_banana_current_scan.py)
- ISHW slide plot packager in [plot_ishw_tradeoffs.py](plot_ishw_tradeoffs.py)
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

There are two wrapper entrypoints for the general ALM workflow:

3. `run_stage2_alm.py`
   This ensures a Stage 2 ALM artifact from either a named built-in profile or a full explicit Stage 2 spec JSON.
   It resolves the full Stage 2 config before launching, pins `--constraint-method alm`, and records the resolved config in the summary output.

4. `run_stage2_to_single_stage.py`
   This is the one-command Stage 2.5 handoff lane.
   It can load or generate a Stage 2 donor, probe Boozer / iota bootability once per donor, optionally run the bounded recovery stage, and then hand off into the full single-stage workflow.

5. `run_single_stage_donor_repair.py`
   This is the batch donor-repair lane.
   It reuses the unified handoff helpers over one or more `--iota-target` values, ranks the resulting donors, and writes a best-donor manifest without launching the full single-stage workflow.

6. `run_stage2_iota_decision_gate.py`
   This is the late-roadmap benchmark / decision-gate lane.
   It runs the same canonical Stage 2 configuration across `report`, `soft`, and `alm` iota modes, records runtime and bootability / iota metrics, and emits a recommendation about whether Stage 2-native iota work is justified.

7. `run_single_stage_thresholded_physics_alm.py`
   This is the generic single-stage ALM rerun lane.
   It requires an explicit Stage 2 artifact plus an explicit plasma surface, pins `--constraint-method alm`, `--alm-formulation thresholded_physics`, and warning-mode hardware handling, and validates that the Stage 2 artifact matches the requested plasma surface before launch.

8. `run_single_stage_goal_mode_comparison.py`
   This is the matched target-vs-frontier comparison lane.
   It requires one explicit Stage 2 artifact plus one explicit plasma surface, runs single-stage twice with identical settings except for `--single-stage-goal-mode {target, frontier}`, validates Stage 2 surface identity, and writes one summary JSON with per-mode metrics plus frontier-minus-target deltas.

There are three wrapper entrypoints for the ISHW analysis / slide-deliverable lane:

9. `run_single_stage_iota_target_sweep.py`
   This reuses one explicit Stage 2 donor and sweeps single-stage `--iota-target` over a user-provided list.
   It writes a compact summary JSON plus a slide-friendly CSV without aborting the entire sweep when one target fails.

10. `run_banana_current_scan.py`
   This reuses one optimized Stage 2 donor, scales the banana-coil current from zero to the donor current, attempts single-stage init-only Boozer startup at each point, and falls back to standalone Poincare artifacts when Boozer startup fails.

11. `plot_ishw_tradeoffs.py`
   This reads the sweep/scan summaries, generates slide-ready figures, and can optionally rerun and copy a reference Poincare directory into one output bundle.

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
├── run_stage2_alm.py
├── run_stage2_to_single_stage.py
├── run_single_stage_donor_repair.py
├── run_stage2_iota_decision_gate.py
├── run_single_stage_iota_target_sweep.py
├── run_banana_current_scan.py
├── plot_ishw_tradeoffs.py
├── run_single_stage_goal_mode_comparison.py
└── run_single_stage_thresholded_physics_alm.py
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

### Stage 2 ALM Wrapper

Use this when you want a general Stage 2 ALM artifact with a small wrapper surface:

- requires `--plasma-surf-filename`
- requires exactly one of `--profile` or `--stage2-spec-json`
- allows a small set of direct CLI overrides for `--cc-threshold`, `--curvature-threshold`, `--order`, and `--tf-current-A`
- writes a compact summary JSON that includes the fully resolved Stage 2 config

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_stage2_alm.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --profile standard_80ka
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_stage2_alm`
- the built-in `standard_80ka` profile now matches the canonical hardware baseline:
  `tf_current_A=8.0e4`, `banana_surf_radius=0.21`, `cc_threshold=0.05`, `curvature_threshold=100`
- the wrapper also exposes the fixed Stage 2 solver-owned clearance contract in its summary/artifact validation:
  `coil-plasma >= 0.015 m`, `plasma-vessel >= 0.04 m`, `coil_length <= 1.7 m`
- `--stage2-spec-json` is the fully explicit path for non-profile Stage 2 contracts
- `--dry-run` prints and records the resolved config and exact Stage 2 command without launching it
- dry runs write `DRY_RUN_ONLY.txt` next to the summary so a summary-only directory is not mistaken for a real solver artifact root
- `--stage2-iota-mode report --stage2-iota-target ...` enables a reporting-only Boozer/iota probe for the generated Stage 2 artifact; this records `BOOZER_BOOTABLE`, `IOTA_FEASIBLE`, `STAGE2_ROOT_FIX_ENABLED`, and `STAGE2_IOTA_*` metadata in `results.json` without changing the Stage 2 optimization objective
- because `run_stage2_alm.py` always launches the Stage 2 solver with `--constraint-method=alm`, its supported iota modes are `off`, `report`, and `alm`; the penalty-path `soft` mode remains available only on the direct `STAGE_2/banana_coil_solver.py` entrypoint

### Unified Stage-2-To-Single-Stage Handoff

Use this when you want the full Stage 2.5 seam in one command:

- accepts either `--stage2-bs-path` or a Stage 2 generation source via `--stage2-profile` / `--stage2-spec-json`
- probes Boozer / iota bootability once per donor before single-stage starts
- optionally runs the bounded recovery stage when the direct donor is not bootable
- writes one summary JSON that records the Stage 2 donor, probe status, recovery outcome, and final single-stage result when invoked in full mode

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_stage2_to_single_stage.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --stage2-bs-path /full/path/to/biot_savart_opt.json
```

Useful notes:

- `--probe-only` stops after the bootability probe
- `--recovery-only` stops after the bounded recovery stage
- `--skip-recovery` keeps this as a pure reporting / handoff classification run
- final single-stage `results.json` is augmented with the shared `BOOTABILITY_*`, `RECOVERY_*`, and `UNIFIED_SEED_SOURCE` provenance fields

### Standalone Donor Repair

Use this when you want to batch the Stage 2.5 repair logic over one or more `iota` targets without launching full single-stage:

- requires the same Stage 2 seed inputs as the unified handoff runner
- accepts one `--iota-target` or a comma-separated `--iota-targets` list
- reuses the exact same probe and recovery helpers as `run_stage2_to_single_stage.py`
- writes a ranked summary plus a `best_repaired_donor.json` manifest when at least one bootable donor is found

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_single_stage_donor_repair.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --stage2-bs-path /full/path/to/biot_savart_opt.json \
  --iota-targets 0.18,0.20,0.22
```

Useful notes:

- `--skip-recovery` ranks direct probe results without attempting bounded recovery
- the best-donor manifest records the selected donor artifact path plus the shared bootability payload
- this entrypoint never launches the full single-stage workflow and rejects `--force-full-single-stage-after-recovery-fail`

### Stage 2 Iota Decision Gate

Use this when you want a reproducible benchmark for the later Track B decision gate:

- runs one canonical Stage 2 configuration across a list of `--benchmark-modes`
- records wallclock, hardware feasibility, bootability, and Stage 2 iota metrics per mode
- computes runtime multipliers versus a baseline mode and emits a recommendation about whether to keep pushing Stage 2-native iota work
- can optionally fold in a prior `run_single_stage_donor_repair.py` summary via `--donor-repair-summary`

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_stage2_iota_decision_gate.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --profile standard_80ka \
  --stage2-iota-target 0.20
```

Useful notes:

- default benchmark modes are `report,soft,alm`
- `--baseline-mode report` compares the hot-loop modes against the probe-only Stage 2 baseline
- the summary CSV is designed for direct spreadsheet / slide-table use when discussing runtime multipliers and iota improvement

### Single-Stage Thresholded-Physics ALM Rerun

Use this when you want a general single-stage ALM rerun from an explicit Stage 2 artifact:

- requires `--plasma-surf-filename`
- requires `--stage2-bs-path`
- forces `--alm-formulation thresholded_physics`
- forces warning-mode hardware handling for ALM because single-surface ALM keeps `gate_scale=1.0`, so adaptive mode would fall back to hard rejection
- validates that the Stage 2 artifact matches the requested plasma surface before launch
- rejects Stage 2 artifacts whose sibling `results.json` reports `init_only=true` unless you explicitly pass `--allow-init-only-stage2-seed`

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --stage2-bs-path /full/path/to/biot_savart_opt.json
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_single_stage_thresholded_physics_alm`
- all `thresholded_physics` thresholds and ALM trust-region settings remain CLI-overridable
- `--dry-run` still validates the Stage 2 artifact metadata and prints the exact single-stage command
- the summary records `stage2_artifact_init_only` from the reused seed metadata when available
- dry runs write `DRY_RUN_ONLY.txt` next to the summary so a summary-only directory is not mistaken for a real single-stage result root

### Single-Stage Goal-Mode Comparison

Use this when you want an A/B comparison between the legacy target-iota objective and the new frontier-iota reward mode from the same Stage 2 seed:

- requires `--plasma-surf-filename`
- requires `--stage2-bs-path`
- launches one `target` run and one `frontier` run under separate output subdirectories
- keeps all other forwarded single-stage settings matched across the two runs, including ALM formulation and thresholds, staged Boozer refinement, multi-surface settings, topology/confinement knobs, plasma current, and banana-surface radius
- always passes `--single-stage-goal-mode` explicitly for both runs, so a parent-shell `SINGLE_STAGE_GOAL_MODE` environment variable cannot silently flip the target/frontier A/B semantics
- rejects Stage 2 artifacts whose sibling `results.json` reports `init_only=true` unless you explicitly pass `--allow-init-only-stage2-seed`
- writes one comparison summary JSON with per-mode metrics, invalid-state reject counts, best-feasible fallback metrics when available, and frontier-minus-target deltas

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_single_stage_goal_mode_comparison.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --stage2-bs-path /full/path/to/biot_savart_opt.json
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_single_stage_goal_mode_comparison`
- `--dry-run` does not require the Stage 2 artifact to exist, but if the artifact and sibling `results.json` are present it still validates the surface match
- the comparison summary records `stage2_artifact_init_only` from the shared seed metadata
- the summary marks `search_objective_values_comparable=false` because `target` and `frontier` intentionally use different base iota terms
- the current `frontier` implementation is `frontier_tradeoff_score_v2`: it uses a seed-normalized tradeoff score with bounded iota/volume rewards, normalized QA/Boozer terms, and a smooth threshold-relative Boozer trust penalty during search while keeping final frontier certification hard
- frontier result payloads record the fixed seed references, effective normalized weights, Boozer trust threshold, and the separate `BOOZER_SURFACE_TARGET_VOLUMES` used by the internal Boozer solve
- when the shared Stage 2 `results.json` includes banana-current metadata, the comparison summary also records the shared seed `BANANA_CURRENT_A` and `BANANA_CURRENT_MAX_A`

### Single-Stage Iota-Target Sweep

Use this for the talk-style tradeoff sweep over increasing `iota` target from one fixed Stage 2 donor:

- requires `--plasma-surf-filename`
- requires `--stage2-bs-path`
- accepts `--iota-targets` as a CSV list
- reuses the same single-stage geometry, Boozer discretization, and hardware settings across every case
- writes one summary JSON plus one slide-friendly CSV

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_single_stage_iota_target_sweep.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --stage2-bs-path /full/path/to/biot_savart_opt.json \
  --iota-targets 0.15,0.18,0.20
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_single_stage_iota_target_sweep`
- failures are recorded per target instead of aborting the full sweep
- `--dry-run` writes only the planned commands plus a summary/CSV bundle

### Banana-Current Scan

Use this for the talk-style scan from zero banana current to the optimized donor current with TF current fixed:

- requires `--plasma-surf-filename`
- requires `--stage2-bs-path`
- accepts `--banana-current-scales` as a CSV list
- mutates the loaded donor banana current after deserialization, so reused donors are scanned correctly even though `--banana-init-current-A` only affects fresh Stage 2 generation
- classifies each point as `success`, `poincare_only_fallback`, or `boozer_failed`

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/run_banana_current_scan.py \
  --plasma-surf-filename wout_nfp10ginsburg_desc_s024match_iota20.nc \
  --stage2-bs-path /full/path/to/biot_savart_opt.json \
  --banana-current-scales 0,0.25,0.5,0.75,1.0
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_banana_current_scan`
- successful init-only single-stage cases reuse their own output directory for Poincare artifacts
- Boozer-init failures still trigger a fallback Poincare artifact path when enough donor metadata exists to rebuild the reference surface

### ISHW Plot Packaging

Use this to turn the sweep and scan summaries into slide-ready plots:

- reads one or both of the summary JSON files from the new wrappers
- can ingest an external field-error versus coil-length CSV or JSON
- can rerun and copy a reference Poincare directory into the final output bundle

```bash
cd /path/to/simsopt-surrogate
python examples/single_stage_optimization/plot_ishw_tradeoffs.py \
  --iota-sweep-summary /full/path/to/single_stage_iota_target_sweep_summary.json \
  --banana-current-scan-summary /full/path/to/banana_current_scan_summary.json
```

Useful notes:

- output root defaults to `examples/single_stage_optimization/outputs_ishw_tradeoffs`
- generated plots include `iota_target` tradeoffs, banana-current tradeoffs, a startup-outcome chart, and a cleaned field-error versus coil-length figure
- the manifest JSON records the exact source summaries plus all generated output paths

## Manual Stage 2

Use [banana_coil_solver.py](STAGE_2/banana_coil_solver.py) when you want to generate or inspect a Stage 2 artifact directly.

Basic penalty-mode example:

```bash
cd /path/to/simsopt-surrogate/examples/single_stage_optimization/STAGE_2
python banana_coil_solver.py \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --major-radius 0.976 \
  --toroidal-flux 0.24 \
  --banana-surf-radius 0.21 \
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
- reporting-only root-fix probe:
  `--stage2-iota-mode report`, `--stage2-iota-target`, `--stage2-iota-tolerance`, `--stage2-iota-vol-target`, `--stage2-iota-constraint-weight` (negative selects exact Boozer Newton mode), `--stage2-iota-num-tf-coils`, `--stage2-iota-nphi`, `--stage2-iota-ntheta`, `--stage2-iota-mpol`, `--stage2-iota-ntor`
- basin-hopping path:
  `--basin-hops`, `--basin-stepsize`, `--basin-temperature`, `--basin-niter-success`, `--basin-seed`

Operational note:

- `--basin-hops` is only supported in penalty mode
- `--constraint-method=alm` and Stage 2 basin-hopping are mutually exclusive in current code

Stage 2 output root layout:

- `STAGE_2/outputs-[plasma_filename]/...`
- the artifact consumed by single-stage is the generated `biot_savart_opt.json`
- when the reporting-only probe is enabled, the sibling `results.json` also records bootability/iota status using the shared Stage 2-to-single-stage contract fields instead of a separate ad hoc schema
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

- the built-in nfp22 single-stage seed defaults now match the current HBT hardware baseline
- in particular, the default nfp22 seed set uses `stage2_seed_tf_current_A = 8.0e4`, `stage2_seed_banana_surf_radius = 0.21`, and `stage2_seed_curvature_threshold = 100`
- if you need a non-baseline seed family, pass the Stage 2 seed / artifact explicitly instead of relying on implicit defaults

## Manual Single-Stage

Use [single_stage_banana_example.py](SINGLE_STAGE/single_stage_banana_example.py) when you want direct control of the single-stage run.

Basic example:

```bash
cd /path/to/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE
python single_stage_banana_example.py \
  --stage2-source database \
  --plasma-surf-filename wout_nfp22ginsburg_000_014417_iota15.nc \
  --stage2-seed-major-radius 0.976 \
  --stage2-seed-toroidal-flux 0.24 \
  --stage2-seed-banana-surf-radius 0.21 \
  --single-stage-goal-mode target \
  --iota-target 0.17 \
  --vol-target 0.10 \
  --cc-dist 0.07 \
  --mpol 15 \
  --ntor 6 \
  --constraint-method penalty
```

ALM operational note:

- single-stage ALM now writes `alm_state.partial.json` inside the run directory at outer-loop transitions and after each recorded ALM history event, so stalled or interrupted runs still leave penalty / multiplier / feasibility diagnostics
- when you are using `--constraint-method alm` in the current single-surface workflow, use `--hardware-search-mode warn` or the dedicated `run_single_stage_thresholded_physics_alm.py` wrapper so trial states can expose constraint violations to ALM instead of being hard-rejected immediately

Current high-level flag groups:

- core problem setup:
  `--plasma-surf-filename`, `--equilibria-dir`, `--equilibrium-path`, `--output-root`, `--mpol`, `--ntor`, `--nphi`, `--ntheta`, `--single-stage-goal-mode`, `--vol-target`, `--iota-target`, `--banana-surf-radius`
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
- the current SIMSOPT API-side conversion is `BOOZER_I = mu0 * PLASMA_CURRENT_A`; the
  literature-side `mu0/(2*pi)` convention is absorbed by SIMSOPT's normalized-angle
  surface parameterization before the `BoozerSurface(..., I=...)` call
- `--constraint-method=alm` currently requires `--num-surfaces=1`
- staged Boozer refinement currently requires penalty mode, single-surface mode, and no basin-hopping
- single-stage basin-hopping is only supported in penalty mode
- `--single-stage-goal-mode=frontier` remains a comparison-first lane, but it now uses a real tradeoff score instead of the earlier iota-only slice
- direct single-stage runs default to `SINGLE_STAGE_GOAL_MODE` from the environment when it is set; the comparison wrapper overrides that by passing the goal mode explicitly
- frontier mode maximizes iota and nested volume with bounded seed-relative rewards, minimizes QA and Boozer residual on normalized scales, and still keeps complexity / buildability penalties in the scalar objective
- Boozer residual plays two roles in frontier mode: it remains a scored metric below the trust threshold, and it becomes an invalid-state reject once the residual exceeds the recorded `FRONTIER_BOOZER_TRUST_THRESHOLD`
- `--vol-target` still feeds the internal Boozer surface construction, but in frontier mode it is no longer reported as the outer optimization target; results instead expose `BOOZER_SURFACE_TARGET_VOLUMES`
- `--iota-target` still seeds the Boozer initialization guess, but frontier mode no longer treats it as the outer optimization target
- for backward-compatible run identities, explicit `--single-stage-goal-mode target` and omitting the flag intentionally hash to the same run fingerprint
- frontier mode rescales legacy `--res-weight` / `--iotas-weight` values relative to their historical defaults before applying the normalized frontier score, so matched target/frontier runs stay in a similar rough magnitude range without reusing the old unbounded `-iota` scalarization
- `--single-stage-goal-mode=frontier` is intentionally incompatible with `--alm-formulation=thresholded_physics` because that ALM formulation still assumes an upper-bounded Jiota penalty objective
- realized hardware status is intentionally split: `search_hardware_status` describes the search-role contract used for accepted/trial single-stage states, while `artifact_hardware_status` describes final artifact certification and drives `HARDWARE_CONSTRAINTS_OK` / `HARDWARE_CONSTRAINT_VIOLATIONS`
- both status objects are schema-driven and now expose `allowed_traversal_status` and `forbidden_traversal_status` buckets so wrappers can distinguish soft-search violations from constraints that are supposed to forbid traversal in penalty mode

## Hardware Search Policy

Search-time realized hardware handling is explicit in current single-stage code:

- `--hardware-search-mode hard`
  Reject hardware-invalid trial candidates during search.

- `--hardware-search-mode warn`
  Keep the search moving, but record realized hardware violations as warnings during search.

- `--hardware-search-mode adaptive`
  Allow warning-only handling only during the early relaxed search window, then revert to hard rejection.

Final certification is still hard in all modes. A terminal hardware-invalid result is still a failure even if search-time handling was softened.

Current constraint semantics are narrower than a generic "all hardware is hard during search" reading:

- in penalty mode, banana current is the only traversal-forbidden search constraint today; it is applied as a hard box bound on the banana-current DOF
- in ALM mode, banana current and coil length are checked again as final feasibility constraints, but temporary search-time violation is still possible
- spacing, curvature, and vessel-clearance constraints are still evaluated through the ordinary search/constraint machinery rather than an always-hard box bound
- `search_hardware_status["forbidden_traversal_status"]` therefore captures the search-role contract, while `artifact_hardware_status` remains the final buildability contract used in result payloads

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
