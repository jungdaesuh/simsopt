# DESC Joint Banana Optimization

This directory is the runner boundary for hardware-first banana-coil co-design.

```text
DESC:
  joint equilibrium + banana-coil optimization

SIMSOPT / this repo:
  seed artifacts, coil group semantics, conversion reports, Boozer/Poincare checks,
  and final hardware/contact oracle status
```

The first executable contracts are preflight, DESC equilibrium runtime-load,
DESC objective assembly/evaluation, conversion-only Lane A smoke, and guarded
Lane A fixed-equilibrium polish plus guarded Lane B/C joint optimizer
execution.
Preflight resolves a hardware spec, seed manifest, equilibrium seed, and DESC
objective stack before any optimizer runs. Runtime-load and objective-assembly
checks prove setup without creating `desc_result.json`. Conversion-only smoke
executes the SIMSOPT -> sampled-DESC-coil -> SIMSOPT bridge and writes loadable
artifacts without claiming DESC optimization. Fixed-equilibrium polish and
joint mode expose the DESC optimizer boundary, but they fail closed by default
before runtime coilset/objective construction because the combined DESC
`ObjectiveFunction` optimizer path has shown high memory use on real banana
seeds. Passing `--allow-high-memory-desc-optimizer` is required before fixed
polish saves `desc_coils.h5` or joint mode saves `desc_equilibrium.h5` plus
`desc_coils.h5`. Physics validation, hardware oracle execution, and promotion
remain blocked until their dedicated launchers bind live evidence to the
exported artifact. The runner deliberately separates these result sections:

- `input_contract`
- `desc_solve_status`
- `search_hardware_status`
- `artifact_hardware_status`
- `physics_validation_status`
- `promotion_status`

## Ownership

DESC owns:

- `BoundaryError` / `VacuumBoundaryError` joint free-boundary objectives.
- `Volume` as the moving-boundary plasma-volume anchor for joint modes.
- DESC coil/equilibrium objects and optimizer calls.
- Generic signed-distance hardware objectives. The banana bridge wires
  `HardwareSdfKeepout` to DESC `CoilSetSDFDistance` when the hardware spec
  provides a manifest-bound `hardware_sdf` artifact.

SIMSOPT owns:

- Stage 2 and single-stage seed generation.
- Banana coil group metadata: TF, banana, proxy, VF, and auxiliary groups.
- Current signs, NFP, symmetry, source checksums, and exported artifact metadata.
- Boozer/Poincare validation.
- Final CAD/contact hardware oracle evidence.

## Failure Policy

The runner fails before optimization when any required binding is missing:

- coil group metadata;
- current signs;
- NFP or stellarator-symmetry provenance;
- source checksums;
- live GLB binding for `hardware_keepout.json` or SDF manifests;
- final oracle path;
- seed surface/field pair;
- LCFS parity between a `simsopt_surface` seed and the constructed DESC surface;
- objective stack rule forbidding `QuadraticFlux` in joint modes.

DESC objective success is not a promotion gate by itself. Promotion requires
SIMSOPT physics validation plus direct loaded-artifact hardware/contact evidence.
That evidence must be bound to the loaded artifact bytes: passed oracle reports
must include `exported_artifact_paths` and `exported_artifact_checksums` matching
the live exported files, plus `source_artifact_checksums` matching the
conversion metadata.

`HardwareSdfKeepout` is an in-loop steering term, not a certification oracle.
DESC samples coil centerlines against the generic SDF objective; the banana
bridge pads the manifest SDF margin by the Type-KK outer-channel corner reach
before constructing the centerline lower bound. Final promotion remains bound
to the SIMSOPT/CAD swept-solid contact oracle.

## Caller Inventory

Affected SIMSOPT surfaces:

- `STAGE_2/banana_coil_solver.py` seed artifacts;
- `SINGLE_STAGE/single_stage_banana_example.py` current single-stage runner;
- goal/frontier comparison tooling under `banana_opt/frontier_*`;
- `POINCARE_PLOTTING` and Boozer validation scripts;
- hardware keepout/oracle code under `banana_opt/hardware_keepout.py`.

Affected DESC surfaces:

- coil classes such as `FourierXYZCoil`;
- objective assembly around `BoundaryError`, `VacuumBoundaryError`,
  `LinkingCurrentConsistency`, `Volume`, `CoilSetMinDistance`, and
  `PlasmaCoilSetMinDistance`;
- optimizer calls that include both the equilibrium and coils;
- free-boundary examples/tests.

## Lane Commands

Lane A preflight:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode fixed_equilibrium_polish \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --output-root /path/to/output \
  --preflight-only
```

Lane B preflight swaps the mode:

```bash
--mode vacuum_joint
```

Lane A conversion-only smoke:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode fixed_equilibrium_polish \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --output-root /path/to/output \
  --conversion-only
```

DESC equilibrium runtime-load check:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode fixed_equilibrium_polish \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --desc-source-root /path/to/DESC \
  --output-root /path/to/output \
  --equilibrium-load-only
```

For `simsopt_surface` equilibrium seeds, this lane loads the SIMSOPT surface or
BoozerSurface wrapper, fits a DESC `FourierRZToroidalSurface`, constructs a
DESC `Equilibrium`, and writes `lcfs_parity` in
`desc_equilibrium_load_report.json`. The parity block records deterministic
sample counts plus max/mean/RMS XYZ deltas between the source LCFS samples and
the fitted DESC LCFS.

DESC objective assembly check:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode fixed_equilibrium_polish \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --desc-source-root /path/to/DESC \
  --output-root /path/to/output \
  --objective-assembly-only
```

DESC objective value smoke:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode fixed_equilibrium_polish \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --desc-source-root /path/to/DESC \
  --output-root /path/to/output \
  --objective-eval-only
```

Real-seed smoke status:

The local bundle `.desc-joint-real-seed-smoke/` uses the real pasted
`winding_ext0009_R0p94_a0p15_FULLPASS` candidate. In this checkout, run it with
`/opt/homebrew/Caskroom/miniforge/base/bin/python3`; the SIMSOPT `.conda-env`
is still the interpreter for SIMSOPT-side contract tests but does not import the
local DESC checkout.

Current real-seed evidence:

- preflight passes and records TF/banana/proxy/VF counts `20/10/0/0`;
- `--equilibrium-load-only` passes with LCFS parity max/mean/RMS XYZ deltas
  `3.8011967608597686e-15` / `1.0338832402145802e-15` /
  `1.2422199144676856e-15` over 441 samples;
- `--objective-assembly-only` passes and writes
  `desc_runtime_coilset_build_report.json` plus
  `desc_objective_assembly_report.json`;
- `--objective-eval-only` passes with sequential per-term value evaluation and
  writes
  `runs/winding_ext0009_objective_eval_sequential/desc_objective_evaluation_report.json`.
  The report has `evaluation_mode: sequential_terms`, `dim_x: 1020`,
  `dim_f: 3762`, finite residuals for all six objective terms, and no Jacobian
  by default. The companion
  `runs/winding_ext0009_objective_eval_sequential/resource_usage_report.json`
  records the measured local max RSS as 10,094,297,088 bytes. The previous
  fused combined-objective value path was SIGKILLed before writing its report.
  Combined Jacobian evaluation remains explicit opt-in via
  `--objective-eval-jacobian`.
- bounded `--fixed-polish-only --desc-optimizer-method scipy-l-bfgs-b
  --desc-maxiter 1` reached DESC optimizer startup but returned after 310.45 s
  with max RSS 12,383,010,816 bytes and macOS peak footprint 43,232,462,688
  bytes at
  `runs/winding_ext0009_fixed_polish_scipy_lbfgsb_maxiter1_watchdog_20260626T175349Z/`;
- default fixed-polish without `--allow-high-memory-desc-optimizer` now fails
  closed before runtime coilset/objective construction and writes
  `runs/winding_ext0009_fixed_polish_default_blocked_20260626T181351Z/desc_result.json`
  plus `desc_fixed_polish_solve_report.json` in 2.81 s with peak footprint
  365,151,600 bytes.

The DESC runtime objective lanes require the paired DESC checkout whose
`LinkingCurrentConsistency` constructor accepts the capped `linking_grid`
keyword. An unpatched DESC install fails closed before objective construction
rather than using the old unbounded linking-current grid path.

Lane A fixed-equilibrium polish default guard:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode fixed_equilibrium_polish \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --desc-source-root /path/to/DESC \
  --output-root /path/to/output \
  --desc-optimizer-method lsq-exact \
  --desc-maxiter 50 \
  --fixed-polish-only
```

This writes a failed `desc_result.json` and solve report before runtime
coilset/objective construction. To actually enter DESC's high-memory optimizer
boundary and produce optimized artifacts, add
`--allow-high-memory-desc-optimizer` in a resource-managed environment.
For continuation/debug runs, the runner exposes only typed DESC optimizer
controls: `--desc-optimizer-ftol`, `--desc-optimizer-xtol`,
`--desc-optimizer-gtol`, `--desc-optimizer-ctol`,
`--desc-optimizer-max-nfev`, and
`--desc-optimizer-min-trust-radius`. These values are forwarded to
`Optimizer.optimize`, recorded in `run_configuration.optimizer.controls`, and
mirrored in the solve report and compact `desc_optimizer_result`.

Lane B/C joint optimizer default guard:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py \
  --mode vacuum_joint \
  --hardware-spec /path/to/desc_joint_hardware_spec.json \
  --seed-manifest /path/to/desc_joint_seed_manifest.json \
  --seed-label slidclean_chomp \
  --equilibrium-seed /path/to/equilibrium_seed.json \
  --desc-source-root /path/to/DESC \
  --output-root /path/to/output \
  --desc-optimizer-method lsq-exact \
  --desc-maxiter 50 \
  --joint-run-only
```

Use `--mode finite_beta_joint` for the finite-beta lane after the vacuum lane
has a validated artifact. Joint mode uses `VacuumBoundaryError` or
`BoundaryError` plus a `Volume` target set to the loaded seed equilibrium
volume through the objective factory; `QuadraticFlux` is not allowed in joint
stacks. The volume anchor prevents the moving LCFS from satisfying the boundary
field objective by inflating away from the seed plasma volume. As with fixed
polish, the default command writes a failed result payload before optimizer
execution. To actually save `desc_equilibrium.h5`, `desc_coils.h5`, and a
loadable SIMSOPT export, add `--allow-high-memory-desc-optimizer` in a
resource-managed environment. Even then, the runner does not self-attest
physics validation, artifact-hardware status, final oracle status, or
promotion.
By default, joint runs use
`--desc-joint-constraint-policy hard-volume-and-force-balance`, so both seed
`Volume` and `ForceBalance` are hard constraints. Methods using DESC's `prox-` /
`proximal-` wrapper are rejected before optimizer execution for that default
path; use a method that directly supports equality constraints such as
`lsq-auglag`.

For a staged proximal probe, use
`--desc-joint-constraint-policy proximal-force-balance`: this moves `Volume`
into the objective stack and leaves only `ForceBalance` as a hard projected DESC
equilibrium constraint for `proximal-lsq-exact`. That is a soft volume target,
not exact volume preservation.

The runner exposes typed DESC proximal-wrapper controls for resource-bounded
debug probes:

- `--desc-proximal-perturb-order`
- `--desc-proximal-solve-maxiter`
- `--[no-]desc-proximal-solve-during-build`

These are serialized to DESC `perturb_options` / `solve_options` and are valid
only with `prox-` / `proximal-` optimizer methods.

Sidecar-consuming SIMSOPT validation wrapper:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/validate_desc_joint_export.py \
  --result /path/to/output/desc_result.json \
  --exported-artifact /path/to/output/biot_savart_desc_export.json \
  --poincare-metrics /path/to/PoincareMetrics_desc_export_validation.json \
  --boozer-state /path/to/surf_desc_export_boozer_state.json \
  --validated-surface /path/to/surf_desc_export_boozer_surface.json \
  --output-root /path/to/output
```

The consumed Poincare/Boozer sidecars must record
`exported_artifact_paths` and `exported_artifact_checksums` for the exact live
exported artifact passed to `--exported-artifact`. This wrapper is physics-only:
it leaves search hardware, exported-artifact hardware, final oracle, and
promotion status blocked until those gates are supplied by their dedicated
oracles. For joint-mode result payloads, the wrapper also requires the
optimized DESC equilibrium artifact from `desc_result.json` and binds the
validated surface to `desc_runtime_artifacts.exported_surface`. If
`--validated-surface` or a Boozer state sidecar's `surface_path` is supplied for
a joint run, it must resolve to that exported DESC surface; otherwise the
wrapper fails closed instead of certifying a stale seed surface.

High-cost SIMSOPT validation launcher:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/launch_desc_joint_validation.py \
  --result /path/to/output/desc_result.json \
  --exported-artifact /path/to/output/biot_savart_desc_export.json \
  --output-root /path/to/output/validation \
  --surface /path/to/validated_surface_for_joint_runs.json \
  --iota 0.09 \
  --G -2.01062
```

The launcher prepares a validation run directory with
`biot_savart_opt.json` and `surf_opt.json`, launches the existing native
`POINCARE_PLOTTING/poincare_surfaces.py` validation mode, re-solves a Boozer
state for the exported field and fixed surface, binds the generated sidecars to
the original exported artifact SHA-256, and then materializes the same physics
validation manifest as `validate_desc_joint_export.py`. Use `--dry-run` to
write only the prepared command/report without launching field-line tracing.
If the Boozer re-solve fails after Poincare metrics are available, the launcher
writes `simsopt_validation_run/surf_desc_export_boozer_state_failed.json` and
continues to materialize a failed physics report/manifest instead of aborting.
Fixed-polish runs may omit `--surface` and default to the selected seed surface.
Fixed-polish Boozer validation reads `input_contract.selected_seed.state` as a
warm start when present; otherwise provide `--iota` and `--G`. Joint runs
default to `desc_runtime_artifacts.exported_surface` and reject any explicit
`--surface` that resolves elsewhere. Joint Boozer validation also requires
explicit `--iota` and `--G`, because seed-state warm starts are not
authoritative after moving-boundary optimization.
This launcher is still physics-only: final artifact hardware and direct
hardware/contact oracle gates remain separate.

Direct hardware/contact oracle launcher:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/launch_desc_joint_hardware_oracle.py \
  --result /path/to/output/desc_result.json \
  --exported-artifact /path/to/output/biot_savart_desc_export.json \
  --oracle-source-artifact /path/to/output/validation/simsopt_validation_run/surf_desc_export_boozer_surface.json \
  --physics-report /path/to/output/validation/desc_joint_simsopt_physics_validation.json \
  --audit-script /path/to/autoresearch/scripts/audit_hardware_contacts.py \
  --output-root /path/to/output/oracle
```

The oracle launcher runs the existing independent
`autoresearch/scripts/audit_hardware_contacts.py` script against an explicit
viewer-exportable source artifact, reads the produced
`hardware_contact_audit.json`, and writes
`desc_joint_final_oracle_evidence.json` only for a zero-hit audit. The evidence
is bound to the original exported artifact SHA-256s and the DESC conversion
source checksums before the validation manifest can pass promotion. For fixed
polish, the recommended oracle source artifact is the BoozerSurface produced by
`launch_desc_joint_validation.py`, not the standalone `BiotSavart` export.
For joint runs, the oracle source artifact must match the validated surface
recorded in the physics report, and that physics report must also bind the live
`desc_equilibrium.h5` checksum from the joint solve.

Joint-mode promotion has one additional dependency: `vacuum_joint` and
`finite_beta_joint` validation manifests include
`fixed_polish_predecessor_status`, and promotion remains blocked unless that
status points to a passed fixed-equilibrium polish validation manifest with
matching source-artifact checksums and strict SIMSOPT physics evidence:
`desc_joint_simsopt_physics_validation_v1`,
`source: simsopt_boozer_poincare_sidecars`, `passed: true`, matching exported
artifact paths/checksums, and live referenced Poincare/Boozer sidecars. The
fixed-polish lane itself is treated as the predecessor lane.

Production outer-loop gate:

```bash
PYTHONNOUSERSITE=1 ./.conda-env/bin/python \
  examples/single_stage_optimization/DESC_JOINT/gate_desc_joint_candidate.py \
  --result /path/to/output/desc_result.json \
  --validation-manifest /path/to/output/oracle/desc_joint_validation_manifest.json \
  --output-root /path/to/output/outer_loop_gate
```

This writes `desc_joint_outer_loop_decision.json`. For `vacuum_joint` and
`finite_beta_joint`, the decision is `accepted` only when the DESC solve,
fixed-polish predecessor, SIMSOPT physics validation, exported-artifact
hardware validation, direct oracle, and promotion status have all passed. Any
missing or failing gate writes `decision: "rejected"` with a `rejection_stage`,
so production search loops can discard violating candidates before advancing.
The gate also verifies that the validation manifest belongs to the same result
payload by comparing exported artifact paths; a candidate/manifest mismatch is
an invalid invocation and exits before writing an acceptance decision.

Current outputs:

- preflight: `desc_joint_preflight.json`
- conversion-only smoke: `desc_result.json`,
  `desc_coils_conversion_only.json`, `biot_savart_desc_export.json`,
  `desc_coil_export_report.json`, `desc_coil_import_report.json`,
  `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md`; `desc_result.json` also records
  `run_configuration`, `run_timing_seconds`, and `run_inventory_path`, and the
  lane writes `desc_joint_run_inventory.json`
- equilibrium runtime-load check: `desc_joint_preflight.json` and
  `desc_equilibrium_load_report.json`
- objective assembly check: `desc_joint_preflight.json`,
  `desc_equilibrium_load_report.json`,
  `desc_runtime_coilset_build_report.json`, and
  `desc_objective_assembly_report.json`
- objective value smoke: the objective assembly outputs plus
  `desc_objective_evaluation_report.json`
- fixed-equilibrium polish default guard: `desc_fixed_polish_solve_report.json`,
  `desc_result.json`, `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md`; `desc_result.json` records
  `run_configuration`, `run_timing_seconds`, and `run_inventory_path`, and the
  lane writes `desc_joint_run_inventory.json`. The solve status is failed, with
  no runtime coilset/objective reports or optimized artifacts unless
  `--allow-high-memory-desc-optimizer` is supplied
- fixed-equilibrium polish with explicit high-memory optimizer opt-in and a
  successful DESC solve: the objective assembly outputs plus
  `desc_fixed_polish_solve_report.json`, `desc_coils.h5`,
  `biot_savart_desc_export.json`, `desc_coil_import_report.json`,
  `desc_optimized_simsopt_export_report.json`, `desc_result.json`,
  `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md`; `desc_result.json` records
  `run_configuration`, `run_timing_seconds`, and `run_inventory_path`, and the
  lane writes `desc_joint_run_inventory.json`. The SIMSOPT export is produced by
  reloading the saved DESC HDF5 `desc_coils.h5` artifact through `desc.io.load`,
  and `desc_optimized_simsopt_export_report.json` records
  `optimized_coilset_source_path`. Artifact hardware, physics validation, and
  promotion remain blocked until the exported SIMSOPT artifact passes the
  dedicated validation/oracle gates
- joint optimizer default guard: `desc_joint_runtime_solve_report.json`,
  `desc_result.json`, `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md`; `desc_result.json` records
  `run_configuration`, `run_timing_seconds`, and `run_inventory_path`, and the
  lane writes `desc_joint_run_inventory.json`. The solve status is failed, with
  no runtime coilset/objective reports or optimized artifacts unless
  `--allow-high-memory-desc-optimizer` is supplied
- joint optimizer run with explicit high-memory optimizer opt-in and a
  successful DESC solve: the objective assembly outputs plus
  `desc_joint_runtime_solve_report.json`, `desc_equilibrium.h5`,
  `desc_coils.h5`, `biot_savart_desc_export.json`,
  `desc_coil_import_report.json`,
  `desc_optimized_simsopt_export_report.json`, `desc_result.json`,
  `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md`; `desc_result.json` records
  `run_configuration`, `run_timing_seconds`, and `run_inventory_path`, and the
  lane writes `desc_joint_run_inventory.json`. The SIMSOPT export is produced by
  reloading the saved DESC HDF5 `desc_coils.h5` artifact through `desc.io.load`,
  and `desc_optimized_simsopt_export_report.json` records
  `optimized_coilset_source_path`. Artifact hardware, physics validation, and
  promotion remain blocked until the exported SIMSOPT artifact passes the
  dedicated validation/oracle gates
- sidecar-consuming validation wrapper:
  `desc_joint_simsopt_physics_validation.json`,
  `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md`
- high-cost validation launcher:
  `desc_joint_simsopt_validation_launch_report.json`,
  `simsopt_validation_run/PoincareMetrics_opt_validation.json`,
  either `simsopt_validation_run/surf_desc_export_boozer_surface.json` plus
  `simsopt_validation_run/surf_desc_export_boozer_state.json`, or
  `simsopt_validation_run/surf_desc_export_boozer_state_failed.json` when the
  Boozer re-solve fails, plus the sidecar-consuming validation wrapper outputs
  after the launched checks finish
- direct hardware/contact oracle launcher:
  `desc_joint_hardware_oracle_launch_report.json`,
  `desc_joint_final_oracle_evidence.json`,
  `desc_joint_validation_manifest.json`, and
  `desc_joint_validation_report.md` when the external swept-solid audit passes
- production outer-loop gate: `desc_joint_outer_loop_decision.json`

Full promotion must not be treated as implemented until the DESC optimizer
artifact has passed SIMSOPT physics validation, exported-artifact hardware
validation, and the direct hardware/contact oracle gate. In-loop DESC-native
hardware SDF steering remains a separate production-hardening item.

`run_configuration` is the canonical per-run operational record for local
memory/performance knobs: resolution preset, DESC source root, DESC evaluation
grid size, Biot-Savart chunk size, distance chunk size, Jacobian chunk size,
DESC and SIMSOPT coil Fourier orders, conversion sample count, optimizer
method, optimizer `maxiter`, optimizer verbosity, objective-eval Jacobian
selection, and selected lane flags. `--resolution-preset smoke` is the default
low-cost local profile; `--resolution-preset production` selects higher
runtime/conversion settings for real runs. Individual resolution flags override
the selected preset and are recorded in the same `run_configuration` block.
`run_timing_seconds` records the phases that actually ran, with absent phases
left as `null`; objective-eval-only timings live in
`desc_objective_evaluation_report.json`.

`desc_runtime_coilset_build_report.json` includes the runtime DESC/SIMSOPT
field parity diagnostic: fixed XYZ probe points, DESC source-grid and chunk
settings, and max/mean field-sample delta between the loaded SIMSOPT
`BiotSavart` and DESC `CoilSet.compute_magnetic_field(..., basis="xyz")`.

For `simsopt_surface` equilibrium seeds, `desc_equilibrium_load_report.json`
includes the runtime DESC/SIMSOPT LCFS parity diagnostic: source and DESC
surface types, deterministic sample counts, source parameter-to-cylindrical
angle drift, and max/mean/RMS XYZ deltas for the fitted DESC LCFS.

For compatibility with existing SIMSOPT artifact consumers, result-producing
lanes also emit legacy hardware/feasibility fields such as
`HARDWARE_CONSTRAINTS_OK`, `HARDWARE_CONSTRAINT_VIOLATIONS`,
`BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK`, and `FINAL_FEASIBILITY_OK`. These are
fail-closed: hardware booleans remain `null` until artifact hardware evidence
exists, `BEST_FEASIBLE_AVAILABLE` is `false`, and `FINAL_FEASIBILITY_OK` is
`false` until the promotion gates pass.

## Troubleshooting

- Current sign mismatch: inspect the seed manifest and conversion report before
  building DESC coils.
- NFP/symmetry mismatch: reject the seed; do not infer from filenames.
- LCFS seed mismatch: inspect `desc_equilibrium_load_report.json` `lcfs_parity`
  before running DESC optimization.
- Hardware provenance failure: regenerate keepout/SDF artifacts from the live GLB.
- Oracle failure: leave `promotion_status` failed or blocked even if DESC solved.
- Oracle binding failure: rerun the hardware/contact oracle on the current
  exported artifact; stale paths or stale SHA-256s are rejected.
- Joint promotion blocked on fixed-polish predecessor: attach a passed
  `fixed_polish_predecessor_status` with live evidence paths from the Lane A
  fixed-equilibrium polish validation before promoting a joint-mode result.
