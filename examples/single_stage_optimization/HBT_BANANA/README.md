# HBT_BANANA Weighted-Only Port

This is an experimental copy of the banana Stage 2 and singlestage workflow for
checking BoozerSurface optimization in this baseline-original checkout.

V1 is weighted-only. It does not import or call an augmented Lagrangian solver,
and it patches only script-level incompatibilities with this SIMSOPT baseline.

## Smoke Commands

From the repository root:

```bash
cd examples/single_stage_optimization/HBT_BANANA
BANANA_STAGE2_MAXITER=20 python jhalpern30/stage2.py 0.0
BANANA_MPOL=2 BANANA_NTOR=2 BANANA_SINGLESTAGE_MAXITER=2 BANANA_SINGLESTAGE_MAXLS=2 python jhalpern30/singlestage.py outputs/I0.0kA/biot_savart_opt.json --current-kA 0.0
```

Use a Python environment with this SIMSOPT baseline and the compiled
`simsoptpp` extension installed; `PYTHONPATH` alone is not sufficient.

Stage 2 writes `biot_savart_opt.json` only when L-BFGS-B reports optimizer
success. Budget-limited or failed runs write `biot_savart_failed.json` and exit
nonzero; failed artifacts are for inspection, not downstream singlestage input.
The singlestage smoke uses low surface resolution so it reaches the L-BFGS-B
path quickly after a successful Stage 2 handoff. A nonzero exit from a Boozer or
SciPy solver failure is expected for this viability check; import/path/API
failures are not.

## Outputs

By default outputs are written under:

```text
examples/single_stage_optimization/HBT_BANANA/outputs/
```

Set `BANANA_OUT_DIR=/path/to/output` to override that directory.

## Registry Lane

The `jhalpern30/` scripts are the direct historical smoke path. They construct
proxy/VF diagnostic coils and are not promotion evidence for the vacuum-current
new-HW campaign.

The top-level `02_stage2_driver.py` and `03_singlestage_driver.py` are the clean
registry lane. They require explicit `warm_start.stage1_id` and
`warm_start.stage2_id` values in `config.yaml`; there is no latest-run fallback.
Their registry metrics include active winding-surface telemetry, poloidal
half-width, vessel clearance, threshold values, signed current telemetry, and
saved Boozer JSON vacuum-lineage checks for Pareto ranking.

## Ported from banana_drivers

Four behaviours were taken from the `banana_drivers` reference package. The
reference's tag-encoded filenames, argparse CLI, and coil-rebuild machinery
were deliberately *not* taken — this lane keeps its content-addressed registry.

- **Boozer-solve failure recovery** (singlestage). A non-converged solve, or
  one landing on a self-intersecting surface, no longer raises. `fun()` returns
  a smooth quadratic reject penalty centered on the last accepted DOFs and
  rolls the surface state back, so the rejected `(J, dJ)` remains a valid
  trial-point pair while L-BFGS-B contracts its line search. Rejected
  evaluations still appear in the diagnostics CSV, flagged by the `solve_ok`
  column (previously hardcoded to `1`).
- **Resolution-aware optimizer tolerances** (singlestage). `ftol`/`gtol` now
  come from `ftol_per_mpol`/`gtol_per_mpol` in `config.yaml`, keyed by
  `boozer.mpol` and clamped outside the tabulated range. At the current
  `mpol=8` these resolve to `1e-5`/`1e-2` — the exact scalars used before, so
  the default run is unchanged.
- **`GlobalRadiusCurvature`** (`new_objectives/global_curvature_radius.py`,
  both stages). Gonzalez–Maddocks self-contact barrier: smooth on its
  noncoincident, nonparallel branch and summed over every near-contact pair,
  with a finite exponential soft penalty. `CurveSelfIntersect` hinges on the
  single worst one and gives no gradient until the constraint is already
  violated. The two enforce the same constraint, so whichever carries a
  nonzero weight owns it and the other is reported as a diagnostic. **This is
  the one behaviour change that moves results**: `singlestage_weights.selfint`
  and `stage2_weights.selfint` are now `0` and `global_curvature` is `1e3`.
  Revert by swapping those two weights back.
  Note its `S_C` divides by `sin^2(alpha)`, not the textbook `2*sin(alpha)`, so
  the reported radius is ~2x the textbook `rho_G` — a circle of radius `R`
  scores exactly `2R`. Read `global_curvature_radius_min` as a calibrated
  barrier threshold, not a physical clearance.
- **Solver stdout capture + iteration tap** (`utils/solver_log.py`,
  singlestage). Folds the Boozer solvers' own stdout into the driver log in
  file order and scrapes the inner BFGS/Newton iteration counts back out, which
  land in the diagnostics CSV as `bfgs_nit`/`newton_nit` and in the registry
  metrics as `boozer_bfgs_nit`/`boozer_newton_nit`.

Adding these changed the registry input whitelist, so **all stage 2 and
singlestage run IDs change**; prior runs are not re-derivable under the new
config. While updating the whitelist, a pre-existing gap was closed:
`thresholds.width_max`/`width_min`/`self_intersect_min` and
`singlestage_weights.width`/`selfint` were never hashed, so editing any of them
previously produced an identical run ID.

Tests live in `tests/geo/test_hbt_banana_{global_curvature_radius,hardware_metrics,run_registry,solver_log}.py`.
The NumPy/stdlib parts run without simsopt; jax-dependent assertions are gated
behind `pytest.importorskip` and require the campaign's FP64 JAX lane for the
central-difference gradient check.
