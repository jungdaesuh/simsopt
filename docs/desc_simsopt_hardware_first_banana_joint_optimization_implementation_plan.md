# DESC/SIMSOPT Hardware-First Banana Joint Optimization Implementation Plan

## Purpose

This file defines the implementation plan for a full hardware-first banana-coil co-design workflow:

```text
DESC:
  optimize equilibrium + banana coils jointly

SIMSOPT / existing repo:
  seed generation, artifact compatibility, Boozer/Poincare checks, hardware oracle
```

The plan is intended to be executable by another engineer or agent without rediscovering the current repo boundaries. It separates confirmed facts from assumptions and keeps final feasibility tied to direct SIMSOPT/CAD artifact validation rather than DESC objective values alone.

## Goals

- Implement a hardware-first workflow that starts from hardware constraints and jointly solves for banana coil geometry, coil currents, and plasma boundary/equilibrium variables.
- Use DESC as the core joint free-boundary optimizer with `BoundaryError` or `VacuumBoundaryError`, not `QuadraticFlux`, for true equilibrium-plus-coil optimization.
- Preserve the existing SIMSOPT banana artifact ecosystem for seed generation, coil group semantics, export compatibility, Poincare/Boozer validation, and hardware oracle checks.
- Add typed, tested conversion contracts between SIMSOPT banana artifacts and DESC equilibrium/coil objects.
- Produce promotion-ready result payloads that distinguish search-time steering, DESC solve status, SIMSOPT physics validation, and final hardware oracle status.

## Non-Goals

- Do not replace the existing SIMSOPT single-stage banana optimizer in one large rewrite.
- Do not treat DESC coil-distance or keepout objective values as a replacement for final CAD/contact/hardware oracle validation.
- Do not use `QuadraticFlux` for joint equilibrium-plus-coil optimization.
- Do not retarget proxy or VF coils as a side effect of adding DESC.
- Do not upstream HBT-specific hardware policy into generic DESC APIs unless the generic abstraction is proven first.

## Current Context

- Current checkout: `/Users/suhjungdae/code/columbia/simsopt-surrogate` on branch `surrogate-confinement-v2`, with pre-existing dirty source/test files. This plan is an additive docs file.
- Current DESC checkout: `/Users/suhjungdae/code/opensource/DESC`.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py` currently describes itself as "single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed" and consumes VMEC/wout-style equilibrium inputs plus SIMSOPT artifacts.
- `examples/single_stage_optimization/banana_opt/single_stage_banana_geometry_mode.py` already owns banana geometry modes: `shared_symmetry`, `materialized_cws`, and `free_xyz`; it preserves coil ordering when replacing banana coils in `BiotSavart`.
- `examples/single_stage_optimization/banana_opt/single_stage_geometry.py` already separates `search_hardware_status` from `artifact_hardware_status`, with final artifact status driving the hardware constraint result fields.
- `examples/single_stage_optimization/banana_opt/hardware_keepout.py` contains SIMSOPT-side keepout machinery including `CurveHardwareKeepout`, `CurveVesselEnvelopeKeepout`, hardware metadata, and hardware keepout loading.
- `examples/single_stage_optimization/banana_opt/hardware_contracts.py` owns current HBT hardware constants, while `examples/single_stage_optimization/banana_opt/hardware_constraint_schema.py` owns current hardware constraint names, thresholds, artifact field names, and payload status fields.
- `examples/single_stage_optimization/banana_opt/artifact_contracts.py` and `examples/single_stage_optimization/banana_opt/stage2_single_stage_handoff.py` already enforce Stage 2 artifact metadata such as `STAGE2_BS_SHA256`, coil counts, `COIL_GROUPS`, current metadata, and hardware artifact fields.
- DESC has coil primitives including `FourierRZCoil`, `FourierXYZCoil`, `CoilSet`, and `MixedCoilSet`.
- DESC has joint/free-boundary objective primitives including `BoundaryError`, `VacuumBoundaryError`, `PlasmaCoilSetMinDistance`, and `LinkingCurrentConsistency`.
- DESC `QuadraticFlux` is explicitly a fixed-equilibrium coil optimization objective. DESC's optimizer rejects `QuadraticFlux` when an `Equilibrium` is in the optimized things.
- DESC supports using saved DESC or VMEC equilibrium output as an initial guess through its existing equilibrium initialization path.

## Rationale

The desired workflow is hardware-first co-design, not coil polish against a fixed surface. The optimization target is:

```text
hardware specs -> allowed coils + allowed plasma boundary -> joint equilibrium/coil solution
```

DESC is the right core for the joint solve because its free-boundary objectives are built around an optimized equilibrium and an external magnetic field. SIMSOPT is the right shell for the banana workflow because the current repo already owns banana-specific CWS geometry, current/group contracts, hardware keepout metadata, artifact formats, and final validation.

Design-it-twice comparison:

- Option A: implement everything inside the current SIMSOPT single-stage script. This minimizes short-term conversion work but keeps the architecture tied to a large script and does not produce a clean DESC free-boundary core.
- Option B: implement a DESC-only banana optimizer. This gives a clean mathematical core but loses current artifact compatibility, hardware oracle integration, and SIMSOPT validation paths.
- Selected option: implement an additive bridge. DESC owns the joint optimizer; SIMSOPT owns seed/artifact/hardware validation. This minimizes replacement risk while creating a path to a true hardware-first solver.

Risk tier:

- The bridge and runner are Tier 2 changes because they introduce new module boundaries and cross-repo contracts.
- Any new exported DESC objective, coil class, or optimizer-facing API is Tier 3 and must include caller inventory, compatibility tests, and rollback notes before being treated as complete.

DESC API evolution note for `CoilSetSDFDistance`:

- Observable behavior delta: `desc.objectives.CoilSetSDFDistance` becomes a new
  exported generic objective that reports one minimum signed regular-grid SDF
  distance per coil and defaults to lower-bound enforcement
  `distance >= minimum_clearance`.
- Caller inventory: the only current in-tree consumer is the
  `HardwareSdfKeepout` entry in
  `examples/single_stage_optimization/banana_opt/desc_bridge/objective_factory.py`;
  existing DESC coil-distance, free-boundary, and optimizer callers are not
  changed.
- Compatibility tests: `DESC/tests/test_objective_funs.py` covers sign
  convention, valid upper-boundary interpolation, patch override, chunked
  evaluation parity, and a finite hard-min descent direction; the
  SIMSOPT-side objective-factory tests cover fail-closed SDF manifest
  requirements and Type-KK centerline padding.
- Migration path: no existing caller must change. New callers must provide
  regular-grid SDF gridsets as `(grid, origin, spacing)` triples, with optional
  patch grids after the base grid.
- Rollback plan: remove the `desc.objectives` export and keep the SIMSOPT
  bridge's final CAD/contact oracle plus post-export hardware gating; promotion
  remains fail-closed without the in-loop SDF steering term.

## Assumptions

- The first implementation can use DESC `FourierXYZCoil.from_values` to represent sampled SIMSOPT banana coils before adding a native DESC banana-CWS curve.
- The first full solve should support vacuum or low-beta free-boundary optimization before finite-beta production runs.
- Existing SIMSOPT `hardware_contracts.py` and `hardware_constraint_schema.py` remain the SSOT for hardware thresholds and result field names; new DESC-joint config may reference them but must not copy their constants into a second schema.
- Existing SIMSOPT hardware keepout JSON/SDF artifacts remain the source for hardware geometry, while final promotion remains tied to the CAD/contact oracle.
- Existing SIMSOPT Poincare/Boozer validation scripts can consume exported DESC-optimized coil artifacts after round-trip conversion.
- VMEC/wout-derived seeds are acceptable initial guesses for DESC, but a filename containing `desc` is not proof of DESC runtime optimization.

## Implementation Plan

Implementation status, 2026-06-28:

- Implemented in the current working tree: the SIMSOPT-side
  contract/preflight slice:
  `DESC_JOINT/README.md`, `DESC_JOINT/hardware_first_schema.md`,
  `DESC_JOINT/run_desc_joint_banana.py`,
  `banana_opt/desc_joint_hardware_spec.py`,
  `banana_opt/desc_joint_seed_manifest.py`,
  `banana_opt/desc_joint_result_schema.py`, and
  `banana_opt/desc_joint_validation.py`.
- Implemented in the current working tree: the first bridge/preflight modules:
  `banana_opt/desc_bridge/coil_export.py`,
  `banana_opt/desc_bridge/coil_import.py`,
  `banana_opt/desc_bridge/conversion_artifacts.py`,
  `banana_opt/desc_bridge/coil_geometry.py`,
  `banana_opt/desc_bridge/equilibrium_seed.py`, and
  `banana_opt/desc_bridge/objective_factory.py`. Joint-mode objective stacks
  include DESC `Volume` targeted to the loaded seed equilibrium volume so the
  moving-boundary solve cannot satisfy the boundary-field objective by
  inflating the LCFS away from the seed plasma volume.
- Implemented in the current working tree: a path-driven SIMSOPT validation
  wrapper, `banana_opt/desc_joint_simsopt_validation.py`, that binds live
  exported artifact SHA-256s to matching Poincare/Boozer sidecar evidence and
  materializes the DESC joint validation manifest via
  `DESC_JOINT/validate_desc_joint_export.py` as a physics-only gate. The CLI
  cannot self-attest search hardware, exported-artifact hardware, final-oracle,
  or promotion status. For joint-mode result payloads, the validated surface is
  bound to `desc_runtime_artifacts.exported_surface`; explicit validated-surface
  paths and Boozer state `surface_path` values must resolve to that exported
  DESC surface or validation fails closed.
- Implemented in the current working tree: a high-cost SIMSOPT validation
  launcher, `banana_opt/desc_joint_validation_launcher.py` plus
  `DESC_JOINT/launch_desc_joint_validation.py`. It prepares a validation run
  directory for a DESC-exported SIMSOPT `BiotSavart`, launches the existing
  native Poincare validation script, re-solves a Boozer state for the exported
  field and validated surface, binds both sidecars to the original exported
  artifact SHA-256s, and then materializes the same physics-only validation
  manifest. It supports `--dry-run` for command/report materialization without
  field-line tracing. It still cannot self-attest exported-artifact hardware,
  final-oracle, or promotion status. For joint-mode result payloads it defaults
  to `desc_runtime_artifacts.exported_surface`, rejects explicit surfaces that
  resolve elsewhere, and requires explicit `--iota` plus `--G` because
  seed-state warm starts are not authoritative after moving-boundary
  optimization.
- Implemented in the current working tree: a direct hardware/contact oracle
  launcher, `banana_opt/desc_joint_hardware_oracle_launcher.py` plus
  `DESC_JOINT/launch_desc_joint_hardware_oracle.py`. It drives the existing
  independent `autoresearch/scripts/audit_hardware_contacts.py` swept-solid
  oracle against an explicit viewer-exportable source artifact, converts only a
  zero-hit audit into `desc_joint_final_oracle_evidence_v1`, binds that evidence
  to live exported artifact SHA-256s and DESC conversion source checksums, and
  writes the final validation manifest/report. For fixed-polish exports the
  expected oracle source artifact is the BoozerSurface emitted by
  `launch_desc_joint_validation.py`, not the standalone `BiotSavart` export.
  For joint-mode result payloads the oracle source artifact must match the
  validated surface in the physics report, and the physics report must bind the
  live `desc_equilibrium.h5` checksum from the joint solve.
- Implemented in the current working tree: an explicit DESC equilibrium
  runtime-load boundary in `banana_opt/desc_bridge/equilibrium_seed.py` and
  `DESC_JOINT/run_desc_joint_banana.py --equilibrium-load-only`. It loads
  `desc_h5` seeds through `desc.io.load`, VMEC `wout` seeds through
  `desc.vmec.VMECIO.load`, constructs DESC `Equilibrium` objects from
  `simsopt_surface` seeds through a sampled LCFS fit to DESC
  `FourierRZToroidalSurface`, writes `desc_equilibrium_load_report.json`, and
  records max/mean/RMS LCFS parity deltas against deterministic source-surface
  samples.
- Implemented in the current working tree: a DESC runtime coilset/objective
  assembly boundary in `banana_opt/desc_bridge/runtime_coilset.py`,
  `banana_opt/desc_bridge/objective_factory.py`, and
  `DESC_JOINT/run_desc_joint_banana.py --objective-assembly-only`. It builds a
  real DESC `CoilSet` from the SIMSOPT seed field through
  `FourierXYZCoil.from_values`, assembles the fixed-equilibrium/joint objective
  stack without optimizing, wires `HardwareSdfKeepout` to DESC
  `CoilSetSDFDistance` when a manifest-bound `hardware_sdf` artifact is present,
  and fails closed when that objective is requested without the SDF manifest.
  Because the generic DESC objective samples coil centerlines, the banana bridge
  converts the swept-surface SDF margin to a conservative centerline bound by
  adding the Type-KK outer-channel corner reach; the SIMSOPT/CAD swept-solid
  contact oracle remains the final certification gate.
- Implemented in the current working tree: a non-optimizer DESC objective
  value smoke lane, `DESC_JOINT/run_desc_joint_banana.py
  --objective-eval-only`, which builds the objective, evaluates scaled
  residuals sequentially per DESC objective term, optionally computes the
  combined Jacobian only when `--objective-eval-jacobian` is explicitly set,
  optionally computes scalar gradients sequentially per objective term when
  `--objective-eval-gradient` is explicitly set, writes an incremental
  `desc_objective_gradient_progress.jsonl` sidecar with term-level RSS snapshots,
  and writes `desc_objective_evaluation_report.json`.
- Implemented in the current working tree: a guarded DESC optimizer execution
  boundary for Lane A fixed-equilibrium polish,
  `DESC_JOINT/run_desc_joint_banana.py --fixed-polish-only`, backed by
  `banana_opt/desc_bridge/runtime_solve.py`. By default it now fails closed
  before runtime coilset/objective construction because real banana DESC
  optimizer artifact export is still an explicitly resource-managed lane, not a
  default smoke path. Passing `--allow-high-memory-desc-optimizer` is required to enter
  `desc.optimize.Optimizer(...).optimize(..., copy=True)`, save the optimized
  DESC `CoilSet` to `desc_coils.h5`, write
  `desc_fixed_polish_solve_report.json`, reload the saved `desc_coils.h5`
  artifact through `desc.io.load`, sample the optimized unique DESC coils
  through `banana_opt/desc_bridge/runtime_export.py`, write the loadable
  SIMSOPT `biot_savart_desc_export.json` plus
  `desc_coil_import_report.json` and
  `desc_optimized_simsopt_export_report.json`, and keep artifact hardware
  validation, physics validation, final oracle status, and promotion blocked.
  Continuation/debug optimizer controls are intentionally typed and narrow:
  `--desc-optimizer-ftol`, `--desc-optimizer-xtol`,
  `--desc-optimizer-gtol`, `--desc-optimizer-ctol`,
  `--desc-optimizer-max-nfev`, and
  `--desc-optimizer-min-trust-radius`. It also exposes typed proximal-wrapper
  controls, `--desc-proximal-perturb-order`,
  `--desc-proximal-solve-maxiter`, and
  `--[no-]desc-proximal-solve-during-build`, which serialize to DESC
  `Optimizer.optimize(..., options={'perturb_options': ..., 'solve_options':
  ...})` and are accepted only for `prox-` / `proximal-` optimizer methods. The
  runner records all of these in `run_configuration.optimizer.controls`, mirrors
  them in solve reports and `desc_optimizer_result`, and passes them directly to
  DESC `Optimizer.optimize`.
- Implemented in the current working tree: a guarded DESC optimizer execution
  boundary for true joint modes,
  `DESC_JOINT/run_desc_joint_banana.py --joint-run-only`, backed by
  `banana_opt/desc_bridge/runtime_solve.py`. It has the same default
  fail-closed optimizer guard; the explicit
  `--allow-high-memory-desc-optimizer` flag is required before it calls
  `desc.optimize.Optimizer(...).optimize(...)` with both the loaded
  `Equilibrium` and DESC `CoilSet` in `things`, saves
  `desc_equilibrium.h5` and `desc_coils.h5`, writes
  `desc_joint_runtime_solve_report.json`, reloads the saved `desc_coils.h5`
  artifact through `desc.io.load`, exports the optimized unique DESC coils
  through the same SIMSOPT `BiotSavart` bridge, and keeps artifact hardware
  validation, physics validation, final oracle status, and promotion blocked.
  The default joint policy,
  `--desc-joint-constraint-policy hard-volume-and-force-balance`, keeps both
  seed `Volume` and `ForceBalance` as hard constraints, so `prox-` /
  `proximal-` DESC wrapper methods are rejected before optimizer execution; that
  path requires a method that directly supports equality constraints such as
  `lsq-auglag`. The explicit hardware-preserving policy,
  `--desc-joint-constraint-policy hard-hardware-and-force-balance`, keeps
  `Volume`, `ForceBalance`, optimized coil current, coil-coil distance,
  plasma-coil distance, coil length, coil curvature, and manifest-bound SDF
  keepout as DESC constraints instead of soft objectives. The explicit staged policy,
  `--desc-joint-constraint-policy proximal-force-balance`, moves `Volume` into
  the objective stack and keeps only `ForceBalance` hard, matching DESC
  `ProximalProjection`'s equilibrium-constraint contract while making volume a
  weighted soft target.
- Implemented in the current working tree: pre-optimizer fail-closed result
  materialization for both opted-in optimizer lanes. After runtime setup passes
  and before calling DESC `Optimizer.optimize`, fixed-polish writes
  `desc_result.json` plus `desc_fixed_polish_solve_report.json`, while joint
  mode writes `desc_result.json` plus `desc_joint_runtime_solve_report.json`.
  If the scheduler or process kills the run inside DESC before the optimizer
  returns, the artifacts left on disk now report `failed` / `blocked` and no
  optimized DESC or SIMSOPT export artifacts.
- Implemented in the current working tree: run-level operational metadata for
  result-producing DESC lanes. `desc_result.json` now records
  `run_configuration` with the DESC source root, evaluation grid, Biot-Savart
  and distance/Jacobian chunk sizes, DESC/SIMSOPT coil Fourier order,
  conversion sample count, optimizer settings, and selected lane flags. It also
  records `run_timing_seconds` for preflight, equilibrium load, coilset build,
  objective assembly, optimizer execution, conversion/export, result
  materialization, and validation-manifest writing where applicable. Each
  result-producing lane writes `desc_joint_run_inventory.json` with absolute
  artifact paths, run mode, objective stack, validation status sections, and
  promotion status.
- Implemented in the current working tree: fail-closed legacy SIMSOPT hardware
  result fields in DESC result payloads. Result-producing lanes now emit
  `HARDWARE_CONSTRAINTS_OK`, `HARDWARE_CONSTRAINT_VIOLATIONS`,
  `BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK`, prefixed artifact hardware fields,
  `BEST_FEASIBLE_AVAILABLE`, and `FINAL_FEASIBILITY_OK` without treating DESC
  setup or steering as final hardware proof.
- Added focused tests:
  `tests/geo/test_desc_bridge_conversion.py`,
  `tests/geo/test_desc_joint_runner_contracts.py`, and
  `tests/geo/test_desc_joint_seed_artifacts.py`.
- Extended the SIMSOPT-side preflight/conversion contract with static
  BiotSavart field inventory, checksum-bound optional source `results.json`
  metadata for `COIL_GROUPS` / legacy counts, auxiliary coil-group preservation,
  source coil names when exposed by the input object, NFP, stellarator
  symmetry, finite-build banana pack metadata, current-sign summaries, sampled
  min-distance deltas, an explicit `coil_conventions` block recording
  source-parameterization preservation, signed-current preservation, and the HBT
  negative-current TF CW convention, live source-artifact checksum revalidation,
  and final-oracle evidence bound to live exported artifact SHA-256s.
- Added a conversion field-sample diagnostic for real SIMSOPT coil inputs:
  the export report compares each original SIMSOPT coil field with its
  Fourier-fit sampled representation reloaded as a SIMSOPT curve at
  deterministic probe points. The round-trip tests compare the full source and
  exported SIMSOPT `BiotSavart` values at fixed probe points. The runtime DESC
  coilset build report now also compares the loaded SIMSOPT `BiotSavart` field
  with the DESC `CoilSet.compute_magnetic_field(..., basis="xyz")` value at a
  fixed probe set and records max/mean field deltas.
- Added the expanded-field DESC runtime coilset contract for SIMSOPT seed
  fields. SIMSOPT `BiotSavart` artifacts in this workflow already materialize
  all physical coils, so the DESC runtime `CoilSet` must not virtualize those
  coils again through `NFP` or stellarator symmetry. Runtime coilset assembly now
  preserves source `nfp` / `stellarator_symmetry` in reports, constructs the
  optimizer field as `NFP=1, sym=False`, records both source and coilset
  symmetry metadata, and fails closed on large DESC/SIMSOPT field-sample parity
  deltas. This corrects the stale Lane B prerequisite-search artifacts whose
  runtime coilset reports showed max/mean field deltas of order `3`-`4` T before
  optimization.
- Added a SIMSOPT-surface LCFS parity diagnostic for equilibrium seeds:
  `desc_equilibrium_load_report.json` includes sample counts, cylindrical
  angle drift from the SIMSOPT parameter grid, and max/mean/RMS XYZ deltas
  between deterministic source-surface samples and the fitted DESC LCFS.
- Added DESC-native LCFS parity diagnostics for non-SIMSOPT equilibrium seeds:
  `desc_h5` seeds now record deterministic loaded-surface self-parity samples,
  while `vmec_wout` seeds compare the loaded DESC LCFS against
  `VMECIO.compute_coord_surfaces` VMEC boundary samples in the runtime load
  report.
- Added an explicit runner resolution policy: `--resolution-preset smoke`
  remains the low-cost local default, `--resolution-preset production` fills
  higher-resolution runtime/conversion settings for real runs, and individual
  resolution flags still override the preset. The selected preset and resolved
  values are recorded in `run_configuration`.
- Added the selected DESC optimized-coil artifact import path: DESC saved-object
  HDF5 (`desc_coils.h5`) is the canonical runtime export format. Fixed-polish
  and joint optimizer lanes now reload `desc_coils.h5` through `desc.io.load`
  before producing the SIMSOPT `BiotSavart` export, and the export report records
  `optimized_coilset_source_path`. DESC MAKEGRID coil text remains a separate
  optional import path, not the selected canonical runtime format.
- Added the fixed-polish-first promotion gate for joint modes: validation
  manifests now carry `fixed_polish_predecessor_status`, and `vacuum_joint` /
  `finite_beta_joint` promotion stays blocked unless a fixed-equilibrium polish
  predecessor status points to a passed fixed-polish validation manifest with
  matching source-artifact checksums and strict SIMSOPT physics evidence:
  `desc_joint_simsopt_physics_validation_v1`,
  `source: simsopt_boozer_poincare_sidecars`, `passed: true`, matching exported
  artifact paths/checksums, and live referenced Poincare/Boozer sidecars.
  Fixed-polish mode itself is treated as the predecessor lane.
- Added a fail-closed constrained outer-loop gate for production joint
  candidates: `banana_opt/desc_joint_outer_loop.py` plus
  `DESC_JOINT/gate_desc_joint_candidate.py` consume `desc_result.json` and the
  checksum-bound validation manifest, require the validation manifest exported
  artifact paths to match the result payload's exported artifact paths, then write
  `desc_joint_outer_loop_decision.json`. The decision accepts only joint
  candidates whose DESC solve, fixed-polish predecessor, SIMSOPT physics
  validation, exported-artifact hardware validation, final oracle, and promotion
  status have passed; failed gates are rejected with a recorded stage, while a
  mismatched result/manifest pair fails before decision materialization.
- Current runner support is preflight, DESC equilibrium runtime-load, DESC
  coilset/objective assembly, objective value smoke with explicit high-memory
  Jacobian opt-in, conversion-only Lane A smoke, guarded fixed-equilibrium
  polish, and guarded joint optimizer execution for `vacuum_joint` /
  `finite_beta_joint`. These runtime objective lanes require
  the paired DESC checkout whose `LinkingCurrentConsistency` constructor accepts
  the capped `linking_grid` keyword; unpatched DESC fails closed before objective
  construction. Runtime
  setup lanes write `desc_equilibrium_load_report.json`,
  `desc_runtime_coilset_build_report.json`, and
  `desc_objective_assembly_report.json` without claiming optimization; the
  optional value/Jacobian smoke also writes
  `desc_objective_evaluation_report.json`. The conversion-only smoke path
  writes `desc_result.json`, sampled DESC-coil JSON,
  `biot_savart_desc_export.json`, conversion reports, and a validation manifest
  while explicitly marking DESC optimization, SIMSOPT physics validation,
  artifact hardware validation, and promotion as not run or blocked. The
  fixed-polish and joint optimizer lanes fail closed by default before runtime
  coilset/objective construction and write failed result payloads unless
  `--allow-high-memory-desc-optimizer` is supplied. With that explicit opt-in,
  both optimizer lanes also pre-write a failed result contract immediately
  before entering DESC `Optimizer.optimize`; a successful DESC solve then
  overwrites that placeholder. Fixed polish writes a real DESC optimizer solve
  report, `desc_coils.h5`, and a loadable exported SIMSOPT `BiotSavart`, while
  the joint lane writes a real DESC optimizer solve report,
  `desc_equilibrium.h5`, `desc_coils.h5`, and a loadable exported SIMSOPT
  `BiotSavart`. Separate launchers can now execute post-export SIMSOPT
  Poincare/Boozer physics validation and the direct hardware/contact oracle for
  those artifacts. The low-resolution Lane B runner path is proven below, but
  no patched Lane B export has a strict SIMSOPT physics-validation pass yet;
  Lane C remains unchecked.
- Promotion manifest helpers now require `desc_solve_status.state == "passed"`
  and final-oracle source checksums matching the expected conversion/source
  checksum map before `promotion_status` can pass.
- Current environment note for DESC-side commands: the SIMSOPT `.conda-env`
  remains the supported interpreter for this repo's SIMSOPT-side contract tests
  but cannot import the local DESC checkout. The Homebrew/miniforge base
  interpreter at `/opt/homebrew/Caskroom/miniforge/base/bin/python3` imports
  DESC, SIMSOPT, JAX, `h5py`, `termcolor`, and `colorama`, and was used for the
  real-seed DESC runner smoke plus the DESC-side pytest commands below.
- Earlier local validation evidence from the first implementation pass:
  `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ./.conda-env/bin/python -m py_compile ...`
  passed for the DESC joint modules/tests; `git diff --check` passed for
  tracked diffs; a trailing-whitespace scan passed over the untracked DESC
  joint files/tests; and
  `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ./.conda-env/bin/python -m pytest -q -p no:cacheprovider tests/geo/test_desc_bridge_conversion.py tests/geo/test_desc_joint_runner_contracts.py tests/geo/test_desc_joint_seed_artifacts.py`
  passed with 91 tests at that checkpoint. Newer Perlmutter contract-suite
  counts are recorded in the Lane B validation bullets below. The local SIMSOPT
  keepout/stage-2 validation command
  `PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ./.conda-env/bin/python -m pytest -q -p no:cacheprovider tests/geo/test_hardware_keepout_winding_frame.py tests/geo/test_stage2_formulation_audit_fixes.py`
  passes with 48 tests and 3 subtests.
- DESC-side validation evidence from the paired DESC checkout:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_objective_funs.py -k 'BoundaryError or VacuumBoundaryError or QuadraticFlux or LinkingCurrentConsistency' -q`
  passes with 4 tests selected and 190 deselected; and
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_optimizer.py -k 'optimize_coil_currents' -q`
  passes with 1 test selected and 88 deselected.
- Current SDF-keepout validation policy after the Perlmutter handoff: do not run
  additional local runtime/JAX/DESC validation for this slice. Perlmutter
  GPU-debug job `55112508` proved CUDA startup on a GPU allocation, then ran the
  bounded DESC SDF unit and SIMSOPT bridge SDF tests on the Perlmutter node after
  forcing CPU execution for validation stability. The same job failed closed
  before real-seed objective construction because the staging hardware spec
  paired the corrected SDF GLB with the old `hardware_keepout.json` point cloud
  whose `provenance.glb_sha256` was stale. That failure is expected contract
  behavior, not an SDF objective failure. The SDF-only resubmission job
  `55114058` again passed CUDA startup plus the bounded DESC/SIMSOPT SDF tests,
  then failed before runner preflight because the selected seed-manifest JSONs
  were not staged at the repo-relative path used by the job script. The missing
  seed JSONs were restored from `/pscratch/.../seed_artifacts/`. Corrected
  GPU-debug resubmission job `55119473` was still pending on scheduler priority
  after the same CUDA-startup path had already been proven, so it was canceled
  before allocation once CPU-debug fallback job `55120328` completed the
  CPU-forced real-seed SDF objective evaluation. Job `55120328` ran on
  Perlmutter `regular_milan_ss11` debug QOS, wrote
  `runs/iota011_R0976_objective_eval_with_sdf_cpu_55120328/`, assembled joint
  objective names `VacuumBoundaryError`, `LinkingCurrentConsistency`,
  `CoilSetMinDistance`, `PlasmaCoilSetMinDistance`, `CoilLength`,
  `CoilCurvature`, and `HardwareSdfKeepout`, evaluated sequential per-term
  values with final term `CoilSetSDFDistance`, reported `dim_x=2891`,
  `dim_f=4623`, `scaled_error_all_finite=true`, and stayed bounded at Slurm
  step max RSS `4905412K`.
- Current Lane D multi-spec evidence: Perlmutter CPU-debug job `55120742`
  completed `COMPLETED 0:0` and wrote
  `runs/lane_d_multispec_summary_55120742.json`. The same
  `iota011_R0976` `vacuum_joint` runner, seed manifest, and equilibrium seed
  evaluated two hardware spec files without Jacobian, gradient, or optimizer
  execution. The base point-cloud keepout spec
  `.desc-joint-real-seed-smoke/hardware_spec.json` assembled/evaluated the
  six-term stack `VacuumBoundaryError`, `LinkingCurrentConsistency`,
  `CoilSetMinDistance`, `PlasmaCoilSetMinDistance`, `CoilLength`, and
  `CoilCurvature` with `dim_x=2891`, `dim_f=4323`, and
  `scaled_error_all_finite=true`. The corrected SDF-only spec
  `.desc-joint-real-seed-smoke/hardware_spec_with_sdf_only_perlmutter.json`
  assembled/evaluated the same runner with additional `HardwareSdfKeepout`
  search steering, evaluated as DESC `CoilSetSDFDistance`, with `dim_x=2891`,
  `dim_f=4623`, and `scaled_error_all_finite=true`. Slurm step max RSS was
  `4157660K` for the base spec and `4115408K` for the SDF-only spec. The mixed
  stale spec that pairs the corrected SDF GLB with the old point-cloud
  `hardware_keepout.json` remains excluded from positive Lane D evidence
  because its purpose is fail-closed provenance checking.
- Current combined-Jacobian evidence: Perlmutter CPU-debug job `55121055`
  completed `COMPLETED 0:0` and wrote
  `runs/winding_ext0009_combined_jacobian_summary_55121055.json` plus
  `runs/winding_ext0009_combined_jacobian_cpu_55121055/desc_objective_evaluation_report.json`.
  The run used `winding_ext0009_R0p94_a0p15_FULLPASS` in
  `fixed_equilibrium_polish` mode, `--objective-eval-only`,
  `--objective-eval-jacobian`, `--no-desc-objective-use-jit`,
  `--desc-objective-deriv-mode blocked`, and banana-only optimized coil
  variables (`dim_x=340`). It reported `dim_f=3762`, Jacobian shape
  `[3762, 340]`, `scaled_error_all_finite=true`,
  `jacobian_all_finite=true`, `jacobian_seconds=269.3061958710896`, and
  Slurm step max RSS `21802760K`. This closes the explicit real-seed
  combined-Jacobian validation for the current banana-scoped contract; it does
  not remove the optimizer guard or imply all-coil dense-Jacobian runs are
  memory-safe.
- Real-seed smoke evidence from the pasted banana candidates: the bundle
  `.desc-joint-real-seed-smoke/` drives the real
  `winding_ext0009_R0p94_a0p15_FULLPASS` artifact from the 2026-06-25 winding
  candidates. Using `/opt/homebrew/Caskroom/miniforge/base/bin/python3`, the
  runner passed preflight, loaded the SIMSOPT Boozer surface as a DESC
  equilibrium with LCFS parity max/mean/RMS XYZ deltas
  `3.8011967608597686e-15` / `1.0338832402145802e-15` /
  `1.2422199144676856e-15` over 441 samples, and assembled the fixed-polish
  objective stack
  `QuadraticFlux`, `LinkingCurrentConsistency`, `CoilLength`, `CoilCurvature`,
  `CoilSetMinDistance`, and `PlasmaCoilSetMinDistance`. After switching the
  smoke lane away from DESC's fused combined-objective value path, the same real
  seed completed `--objective-eval-only` and wrote
  `runs/winding_ext0009_objective_eval_sequential/desc_objective_evaluation_report.json`
  with `evaluation_mode: sequential_terms`, `dim_x: 1020`, `dim_f: 3762`, and
  finite residuals for all six objective terms before optimizer-variable
  scoping landed. The companion
  `runs/winding_ext0009_objective_eval_sequential/resource_usage_report.json`
  records the `/usr/bin/time -l` runtime and max RSS. The previous fused
  combined-objective path was SIGKILLed before writing its evaluation report.
  The same real NFP=5 banana surface was also materialized as generated DESC H5
  and VMEC wout seed-source artifacts at
  `generated_equilibrium_sources/winding_ext0009_desc_equilibrium_seed.h5` and
  `generated_equilibrium_sources/wout_winding_ext0009_desc_equilibrium_seed.nc`.
  Loading those generated sources through the runner with
  `--equilibrium-load-only` passed at
  `runs/winding_ext0009_desc_h5_equilibrium_load/desc_equilibrium_load_report.json`
  and
  `runs/winding_ext0009_vmec_wout_equilibrium_load/desc_equilibrium_load_report.json`.
  The DESC H5 report records 441 LCFS parity samples with zero max/mean/RMS XYZ
  delta, and the VMEC wout report records 441 VMEC-boundary parity samples with
  max/mean/RMS delta `7.906990629795298e-16` /
  `2.9251416658690877e-16` / `3.267052015784014e-16` m.
  A post-import-helper rerun also passed at
  `runs/winding_ext0009_objective_eval_sequential_rerun_import_fix/` with
  `evaluation_mode: sequential_terms`, `dim_x: 1020`, `dim_f: 3762`, and a
  recorded max RSS of `8446361600` bytes. Combined Jacobian evaluation is still
  a high-memory explicit opt-in and is not proven on this real seed.
  After reducing the optimizer-facing DESC `CoilSet` to the banana group while
  keeping all 30 coils in field/objective evaluation, the current real-seed
  value smoke passed at
  `runs/winding_ext0009_objective_eval_banana_scope_fullparam_miniforge_20260626T181944/`
  with `evaluation_mode: sequential_terms`, `dim_x: 340`, `dim_f: 3762`, and a
  scope report showing 10 optimized banana coils plus 20 fixed TF/proxy/VF
  coils.
  The real seed also passed the conversion-only Lane A bridge/export path at
  `runs/winding_ext0009_conversion_only_import_fix/`: the runner wrote
  `desc_result.json`, `desc_coils_conversion_only.json`,
  `biot_savart_desc_export.json`, `desc_joint_validation_manifest.json`,
  `desc_joint_validation_report.md`, and `desc_joint_run_inventory.json` in
  `14.38` seconds with max RSS `749649920` bytes. Loading the exported artifact
  through the repo SIMSOPT environment produced a
  `simsopt.field.biotsavart.BiotSavart` with 30 coils. This proves the
  real-seed bridge/export/result-schema path without claiming DESC optimization,
  physics validation, hardware oracle success, or promotion.
  A bounded Lane A optimizer attempt using `--fixed-polish-only`,
  `--desc-optimizer-method lsq-exact`, and `--desc-maxiter 1` reached preflight,
  equilibrium load, coilset build, and objective assembly, but did not write
  `desc_fixed_polish_solve_report.json`, `desc_result.json`, `desc_coils.h5`,
  or `biot_savart_desc_export.json` after `39:30` elapsed. It was terminated
  with `SIGTERM`; the blocker evidence is recorded in
  `runs/winding_ext0009_fixed_polish_maxiter1_import_fix/lane_a_fixed_polish_timeout_resource_report.json`.
  A process sample showed the main thread blocked in
  `jax::PyArray::BlockUntilReady` while XLA CPU worker threads executed kernels,
  so the remaining Lane A blocker is the exact optimizer/JAX derivative path,
  not objective assembly or the value-only smoke evaluator.
  A follow-up bounded optimizer attempt using `--fixed-polish-only`,
  `--desc-optimizer-method scipy-l-bfgs-b`, and `--desc-maxiter 1` reached
  DESC optimizer startup but returned `rc=1` after `310.45` seconds with max RSS
  `12383010816` bytes and macOS peak memory footprint `43232462688` bytes at
  `runs/winding_ext0009_fixed_polish_scipy_lbfgsb_maxiter1_watchdog_20260626T175349Z/`.
  A Perlmutter GPU-debug rerun exposed and fixed a separate launch blocker:
  DESC defaults to CPU when `desc.backend` is imported before
  `desc.set_device("gpu")`, which clears the CUDA device selection. The runner
  now supports early DESC device bootstrap through `--desc-runtime-device` or
  `DESC_JOINT_DESC_DEVICE` before importing DESC bridge modules. With
  `--desc-runtime-device gpu`, Perlmutter job `55098630` reached CUDA-backed
  equilibrium load, coilset build, objective assembly, and one
  `scipy-l-bfgs-b` optimizer iteration. It still failed to export an optimized
  artifact because DESC returned `optimizer_success: false`, status `99`, and
  message ``callback` raised `StopIteration`` after `nfev: 4`, `nit: 1`; Slurm
  recorded max RSS `43246788K`. Follow-up Perlmutter GPU-debug runs after
  banana-only optimizer scoping and top-level DESC `ObjectiveFunction`
  no-JIT/blocked construction still reached about `43`-`44` GB RSS with
  `scipy-l-bfgs-b` and returned failed optimizer status `99` without export.
  Source audit showed `lsq-exact` materializes the dense full Jacobian and is not
  the low-memory path; the corrected `lsq-exact` GPU-debug probe `55101083` was
  canceled after that audit. Perlmutter GPU-debug objective-gradient jobs
  `55102191` and `55102355` then ran `--objective-eval-only
  --objective-eval-gradient` successfully on the same seed. Job `55102355`
  recorded Slurm max RSS `43369756K`; the progress sidecar proves RSS stayed at
  `16845060` ru_maxrss units through `QuadraticFlux`,
  `LinkingCurrentConsistency`, `CoilLength`, and `CoilCurvature`, then jumped to
  `43450888` during the `CoilSetMinDistance` scalar-gradient term. A first
  DESC-only patch did not affect the active smoke evaluator: Perlmutter job
  `55103434` still recorded Slurm max RSS `43363780K` because
  `_compute_scalar_gradient_by_term` was manually differentiating
  `objective.compute_scaled_error(...)`. After switching that active evaluator
  to differentiate DESC `objective.compute_quadratic_scalar(...)`, and after
  adding a DESC memory-bounded hard-min scalar-gradient path for
  `CoilSetMinDistance`, Perlmutter job `55103718` passed the same
  `--objective-eval-only --objective-eval-gradient` probe with Slurm max RSS
  `16744128K`. Its progress sidecar stayed flat at `16869808` ru_maxrss units
  through every term, including `CoilSetMinDistance`, which finished in
  `5.1625439340714365` seconds. The memory-heavy coil-coil scalar-gradient
  blocker is therefore closed for the blocked/no-JIT scalar-gradient lane.
  A follow-up Perlmutter fixed-polish optimizer smoke, job `55104085`, used
  `--fixed-polish-only --allow-high-memory-desc-optimizer
  --desc-optimizer-method scipy-l-bfgs-b --desc-maxiter 1` with the same
  blocked/no-JIT objective policy. It completed in `00:02:23` with Slurm max RSS
  `16768408K`, passed objective assembly, and wrote
  `desc_fixed_polish_solve_report.json` plus `desc_result.json`; the solve
  report is `status: failed` only because DESC did not report `success=True`
  under `maxiter: 1`, so no `desc_coils.h5` or SIMSOPT export was produced.
  A maxiter-10 rerun, Perlmutter job `55104808`, reached DESC optimizer
  success (`optimizer_success: true`, `optimizer_status: 0`, `optimizer_nit: 1`,
  `optimizer_nfev: 5`) and wrote `desc_coils.h5`, but exposed an export-shape
  bug: saved DESC `FourierXYZCoil` artifacts can return native samples as
  `(1, 129, 3)` despite a requested 64-point grid. The SIMSOPT export bridge now
  squeezes singleton leading dimensions, resamples periodic XYZ traces to the
  requested `sample_count`, and then validates the final `(64, 3)` contract. A
  local export replay from job `55104808` wrote
  `runs/winding_ext0009_fixed_polish_local_exportfix_from_55104808/biot_savart_desc_export.json`
  with a passed export report. The patched Perlmutter end-to-end rerun, job
  `55105011`, completed `COMPLETED 0:0` with Slurm max RSS `16810932K`, passed
  fixed-polish solve/export, wrote `desc_coils.h5` and
  `biot_savart_desc_export.json`, and the exported artifact loads in the local
  SIMSOPT environment as a `BiotSavart` with 30 coils. Lane A fixed-polish
  optimizer/export is therefore proven for this real seed. Remaining proof is
  downstream validation/promotion: SIMSOPT Poincare/Boozer physics validation,
  exported-artifact hardware oracle, and final promotion remain fail-closed.
  The resolved blockers were CUDA startup, objective assembly, value-only
  evaluation, full-CoilSet variable exposure, coil-coil scalar-gradient memory,
  optimizer convergence for the real seed, and DESC-H5-to-SIMSOPT export shape.
  Therefore fixed-polish and joint optimizer lanes now fail closed by default
  unless `--allow-high-memory-desc-optimizer` is supplied. The real seed default
  guard was verified at
  `runs/winding_ext0009_fixed_polish_default_blocked_20260626T181351Z/`: it
  wrote `desc_result.json` and `desc_fixed_polish_solve_report.json` in `2.81`
  seconds with max RSS `441712640` bytes and peak footprint `365151600` bytes,
  with no runtime coilset/objective reports or optimized DESC/SIMSOPT artifacts.
  The broader pasted seed inventory is now anchored at
  `/Users/suhjungdae/code/columbia/autoresearch/campaigns/balance_pareto_singlestage_2026-06-17/runs/poincare_default_gallery_2026-06-25/DIAGNOSTICS_full_2026-06-25.csv`.
  It contains 15 labeled candidates with iota, volume, aspect, residual,
  curvature, coil-count, and current diagnostics. The matching
  `PoincareDefault_<label>.json` sidecars in the same gallery directory provide
  the surface and field paths for constructing follow-on seed manifests. The
  current 15-candidate DESC runner manifest lives at
  `.desc-joint-real-seed-smoke/seed_manifest_poincare_gallery_15.json`; it
  contract-validates all 15 labels, keeps the m36 entries as `bare_surface`
  seeds with explicit matching fields, and records explicit 20 TF / 10 banana
  coil groups for candidates without source `results.json` metadata.
  A first selected gallery sweep staged remote-path seed artifacts for
  `iota011_R0976`, `iota011_R0935`, and
  `winding_band0015_R0p92_a0p15` under
  `/pscratch/sd/j/jungdae/desc_joint_debug_20260626T170233/seed_artifacts/poincare_gallery_selected/`
  and ran Perlmutter debug-GPU job `55105529`. The job completed
  `COMPLETED 0:0` in `00:06:36`; the three fixed-polish/export steps completed
  in `00:02:09`, `00:02:06`, and `00:02:06` with Slurm MaxRSS `15969068K`,
  `16556328K`, and `16563036K`. All three runs report
  `optimizer_success: true`, `optimizer_status: 0`, `optimizer_nit: 1`,
  `optimizer_nfev: 5`, export `sample_count: 64`, and load locally as 30-coil
  SIMSOPT `BiotSavart` artifacts. Pulled run directories live under
  `.desc-joint-real-seed-smoke/runs/*_gallery3_55105529/`.
  Post-export physics validation was then run for all three selected gallery
  exports. Each re-solved a Boozer state successfully (`iota` shifted to
  `0.1205`-`0.1248`, residuals at machine precision), but each strict
  validation-mode Poincare sidecar recorded `fails_validation` with `42/50`
  surviving lines, so `desc_joint_simsopt_physics_validation.json` records
  `passed: false` and promotion remains failed. This is stricter than the
  earlier default-gallery exploratory plots, which used default-mode
  extended-surface seeding and reported healthy `42`-`43/50` survival; strict
  validation mode requires `50/50` survival. The direct hardware/contact oracle
  was run for the `iota011_R0976` exported BoozerSurface and passed: the audit
  reported `hardware contacts: 0/10 -- CLEAR`, wrote
  `desc_joint_final_oracle_evidence.json`, and the combined validation manifest
  records `artifact_hardware_status.passed: true` plus
  `final_oracle_status.passed: true`; `promotion_status` still correctly fails
  because physics validation did not pass.
  A first true Lane B vacuum-joint Perlmutter debug-GPU solve was run for
  `iota011_R0976` as job `55107392`. It completed `COMPLETED 0:0` in
  `00:04:59` with Slurm step MaxRSS `15610376K`, reached DESC optimizer
  success (`scipy-l-bfgs-b`, `nit: 4`, `nfev: 19`, final objective
  `1.861e+01`), wrote `desc_equilibrium.h5`, `desc_coils.h5`,
  `biot_savart_desc_export.json`, and the moved plasma-boundary export
  `surf_desc_equilibrium_export.json`. The exported surface loads as
  `SurfaceXYZTensorFourier(nfp=5, mpol=10, ntor=10, stellsym=True)` and the
  surface export report records max/RMS fit residuals
  `2.6854146949332733e-15` / `8.699689039413326e-16` m. Strict SIMSOPT
  validation of that moved-boundary joint export was rerun on Perlmutter as job
  `55108116`; four focused validation-regression tests passed remotely, the
  launcher completed without crashing, and
  `simsopt_validation_strict_failed_boozer_evidence_55108116/desc_joint_simsopt_physics_validation.json`
  records `passed: false`. The strict Poincare sidecar is
  `fails_validation` with `0/50` surviving lines and `surface_exit: 50`; the
  Boozer re-solve is preserved as
  `simsopt_validation_run/surf_desc_export_boozer_state_failed.json` with
  `passed: false`, `iter=40`, `iota=-640.4404149117591`,
  `G=-4.334966828644297e-05`, and `residual_inf=1.102e-10`. The validation
  manifest now reports `promotion_status.state: failed` with reason
  `SIMSOPT physics validation did not pass`. This proves the Lane B
  solve/export/report path and the failed-validation evidence path, but it is
  not a promotion-ready candidate.
  A follow-up Lane B prerequisite search on Perlmutter GPU-debug job `55121657`
  tried the staged `iota011_R0935` and
  `winding_band0015_R0p92_a0p15` gallery seeds. Both DESC vacuum-joint solves
  completed successfully (`optimizer_success: true`, `nit: 5` and `4`), but
  strict SIMSOPT validation failed for both with `0/50` surviving Poincare
  field lines. The summary
  `.desc-joint-real-seed-smoke/runs/lane_b_prereq_search_summary_55121657.json`
  records `status: no_passing_candidate`, so no artifact currently satisfies
  the Lane C prerequisite. A later field-parity audit found those Lane B runs
  used stale runtime-coilset semantics that over-virtualized already-expanded
  SIMSOPT coils. After syncing the expanded-field `NFP=1, sym=False` runtime
  coilset contract to Perlmutter, GPU-debug job `55148944` passed objective
  assembly for `winding_band0015_R0p92_a0p15` with
  `coilset_nfp: 1`, `coilset_stellarator_symmetry: false`,
  `source_nfp: 5`, `source_stellarator_symmetry: true`,
  `max_delta_T: 0.016951885063729997`, and
  `mean_delta_T: 0.005167423993795285`. The patched Lane B optimizer rerun is
  therefore required before treating the earlier `0/50` strict-validation result
  as a final physics verdict for that seed. CPU-debug fallback job `55150212`
  then completed the patched `winding_band0015_R0p92_a0p15` Lane B
  optimizer/export/strict-validation path in `00:02:42` with Slurm step max RSS
  `5215324K`: DESC optimization passed (`optimizer_success: true`, `nit: 3`,
  `nfev: 6`), the exported `BiotSavart` and moved surface were materialized,
  field parity stayed fixed (`max_delta_T: 0.016951885063729997`,
  `mean_delta_T: 0.005167423993795285`), but strict SIMSOPT validation still
  failed with `0/50` surviving field lines, all `surface_exit`, and promotion
  failed with reason `SIMSOPT physics validation did not pass`.

1. Establish contracts and ownership
   - [x] Create `examples/single_stage_optimization/DESC_JOINT/README.md` documenting the selected architecture: DESC joint solver, SIMSOPT seed/export/validation shell.
   - [x] Define a typed result schema for hardware-first joint runs with separate sections for `input_contract`, `desc_solve_status`, `search_hardware_status`, `artifact_hardware_status`, `physics_validation_status`, and `promotion_status`.
   - [x] Define the failure policy for missing artifact binding fields such as coil group metadata, current signs, NFP, symmetry, source checksums, and hardware keepout provenance.
   - [x] Add a caller inventory for affected SIMSOPT entrypoints: Stage 2 solver, current single-stage runner, goal-mode comparison runner, frontier trajectory tooling, and Poincare/Boozer validation scripts.
   - [x] Add a caller inventory for affected DESC surfaces: coil classes, objective assembly, optimizer calls, and free-boundary examples/tests.

2. Extend the existing hardware-contract SSOT for DESC joint runs
   - [x] Add `examples/single_stage_optimization/DESC_JOINT/hardware_first_schema.md` with required runner inputs: vessel/port keepout sources, coil length limits, coil-coil spacing, coil-plasma spacing, curvature limits, current limits, width limits, TF/proxy/VF policy, and final oracle path.
   - [x] In `hardware_first_schema.md`, map every threshold and status field back to `banana_opt/hardware_contracts.py` or `banana_opt/hardware_constraint_schema.py`; do not introduce a second set of hardware constants.
   - [x] Add `examples/single_stage_optimization/banana_opt/desc_joint_hardware_spec.py` with frozen typed dataclasses for DESC-joint runner configuration. The dataclasses should reference the existing hardware schema and carry paths/provenance, not duplicate threshold values.
   - [x] Add a loader that reads JSON hardware specs and existing `hardware_keepout.json`/SDF metadata without parsing Markdown as configuration.
   - [x] Add validation that fails loudly when hardware geometry provenance is missing, stale, or not bound to the GLB/SDF artifact used by the run.
   - [x] Add unit tests proving DESC-joint config resolves the current hard constraints through the existing schema: max coil length, min coil-coil spacing, max curvature, coil-plasma spacing, banana current cap, TF current cap, width bounds, and keepout source identity.

3. Build SIMSOPT to DESC conversion
   - [x] Add `examples/single_stage_optimization/banana_opt/desc_bridge/coil_export.py` to convert SIMSOPT coils into DESC coil objects.
   - [x] Preserve coil groups explicitly: TF, banana, proxy, VF, and any loaded auxiliary groups.
   - [x] Preserve current values in Amperes, current signs, source coil names when exposed by the input object, ordering, NFP, symmetry, and finite-build banana pack metadata in conversion reports.
   - [x] Convert banana CWS curves by sampling dense Cartesian coordinates and fitting DESC `FourierXYZCoil` objects at a configured Fourier order.
   - [x] Record coordinate basis, sample count, fit residual, length delta, curvature delta, min-distance delta, and field-sample delta in a conversion report.
   - [x] Add tests that fail if CWS materialization or free-XYZ conversion changes coil ordering or current signs.
   - [x] Add tests that compare SIMSOPT and DESC magnetic fields at a fixed set of points within a tolerance chosen from observed conversion residuals.
   - [x] Add a runtime field-parity guard for real SIMSOPT seed fields that
     treats source `BiotSavart` artifacts as expanded physical coil sets:
     runtime DESC coilsets use `NFP=1, sym=False`, reports preserve both source
     and coilset symmetry metadata, and stale symmetry over-virtualization fails
     closed before objective construction.

4. Build DESC to SIMSOPT conversion
   - [x] Add `examples/single_stage_optimization/banana_opt/desc_bridge/coil_import.py` to convert sampled DESC-coil bridge outputs back to SIMSOPT-compatible `BiotSavart` artifacts.
   - [x] Add the import path for real DESC optimized coil artifacts or MAKEGRID artifacts once the canonical DESC runtime export format is selected. DESC saved-object HDF5 (`desc_coils.h5`) is the selected canonical runtime export format; MAKEGRID coil text remains a separate optional path if a future run needs that external interchange format.
   - [x] Preserve the original coil group manifest during import so downstream validation does not infer groups from heuristics.
   - [x] Add round-trip tests: SIMSOPT -> DESC -> SIMSOPT should preserve sampled coordinates, current signs, lengths, min distances, field samples, and group labels within explicit tolerances.
   - [x] Add artifact metadata fields for DESC optimizer version, DESC commit, conversion residuals, and source artifact checksums.
   - [x] Add a fail-closed check that prevents validation from treating a DESC-exported artifact as hardware-clean until the existing hardware oracle has run.

5. Seed DESC equilibrium and plasma boundary state
   - [x] Add `examples/single_stage_optimization/banana_opt/desc_bridge/equilibrium_seed.py` to validate explicit DESC equilibrium seed provenance specs for DESC, VMEC/wout, or SIMSOPT-surface sources.
   - [x] Load actual DESC `Equilibrium` seed objects from DESC H5 and VMEC/wout inputs through an explicit runtime-load lane.
   - [x] Add the SIMSOPT-surface-to-DESC `Equilibrium` construction path instead of treating `simsopt_surface` provenance as directly loadable.
   - [x] Preserve NFP, handedness, stellarator symmetry, angular convention, major/minor scale, and LCFS mode truncation choices in a seed report.
   - [x] Add an equilibrium runtime-load smoke path that proves DESC can load the declared seed before enabling optimization.
   - [x] Add a fixed-equilibrium smoke path that loads the seeded equilibrium and assembles DESC coil objectives against a runtime DESC coilset before enabling optimization.
   - [x] Add a fixed-equilibrium smoke path that evaluates DESC coil objective values and gradients against the seeded equilibrium before enabling joint optimization in synthetic tests. The real banana seed now has passing sequential objective-value, sequential per-term scalar-gradient, and explicit opt-in combined-Jacobian smoke evidence. The Jacobian path remains high-memory and non-default.
   - [x] Add tests comparing seeded LCFS samples against the source VMEC/SIMSOPT surface at deterministic grid points.
   - [x] Add a guard that marks the seed provenance as `vmec_wout`, `desc_h5`, or `simsopt_surface`; do not infer DESC provenance from filenames.

6. Implement DESC objective assembly
   - [x] Add `examples/single_stage_optimization/banana_opt/desc_bridge/objective_factory.py` for assembling DESC objectives from typed config.
   - [x] Implement a fixed-equilibrium coil-polish objective mode using `QuadraticFlux`, `LinkingCurrentConsistency(eq_fixed=True)`, coil length, coil curvature, coil-coil distance, and plasma-coil distance.
   - [x] Add runtime assembly/reporting for DESC `BoundaryError` or `VacuumBoundaryError`, `LinkingCurrentConsistency(eq_fixed=False)`, `PlasmaCoilSetMinDistance(eq_fixed=False)`, coil length, coil curvature, and coil-coil distance.
   - [x] Implement the true in-loop hardware keepout term for joint mode. Source wiring maps `HardwareSdfKeepout` to DESC `CoilSetSDFDistance` with manifest-bound SDF provenance and Type-KK centerline padding. Perlmutter jobs `55112508` and `55114058` proved CUDA startup plus bounded unit/bridge gates on the allocation while failing closed on non-SDF staging issues; CPU-debug fallback job `55120328` then passed real-seed SDF-only joint objective assembly/value evaluation with `HardwareSdfKeepout` in the joint stack and `CoilSetSDFDistance` in the evaluated term reports.
   - [x] Add a test that joint mode rejects `QuadraticFlux` before reaching the DESC optimizer.
   - [x] Add objective scaling metadata so reported residuals can be compared across fixed-equilibrium polish, vacuum joint solve, and finite-beta joint solve.
   - [x] Add chunk-size configuration only for compute/memory infrastructure concerns; do not expose algorithm-choice knobs as user-facing switches unless they are necessary for external operational constraints.

7. Add generic in-loop hardware objectives for production
   - [x] For the first Lane A smoke test, existing DESC coil/plasma distance objectives plus SIMSOPT post-validation are acceptable because the lane is a conversion and fixed-equilibrium polish check, not a hardware-first production claim.
   - [x] Add a fail-closed constrained outer-loop gate for Lane B/C candidates that rejects and records hardware/physics/oracle failures before promotion. Finite-beta promotion additionally requires an explicit passed Lane B `vacuum_joint` predecessor validation manifest through `lane_b_predecessor_status`.
   - [x] Add true search-time hardware steering beyond post-export gating if production exploration needs hardware constraints to influence candidate generation before export. The joint objective stack now includes `HardwareSdfKeepout`; job `55120328` proves the real `iota011_R0976` seed assembles and evaluates that term before export/promotion gates.
   - [x] If direct hardware keepout must be differentiable inside DESC, add generic objective support in DESC rather than HBT-specific names. The paired DESC checkout implements the generic `CoilSetSDFDistance` objective.
   - [x] Candidate DESC implementation: `desc/objectives/_hardware.py` with a regular-grid signed-distance input and coil-to-SDF objective.
   - [x] Export new objective(s) through `desc/objectives/__init__.py` only after unit tests and API-evolution notes exist. `CoilSetSDFDistance` is exported and documented in the DESC API evolution note above.
   - [x] Add DESC tests with synthetic SDF fixtures proving sign convention, boundary interpolation, patch override, chunking behavior, and finite descent direction for the hard-min gradient. The synthetic DESC test passed on the Perlmutter allocation in jobs `55112508` and `55114058`.
   - [x] Keep the SIMSOPT CAD/contact oracle as the final promotion gate even if DESC gets differentiable keepout steering.

8. Implement the DESC joint runner
   - [x] Add `examples/single_stage_optimization/DESC_JOINT/run_desc_joint_banana.py`.
   - [x] Support modes: `fixed_equilibrium_polish`, `vacuum_joint`, and `finite_beta_joint`.
   - [x] Load hardware spec, SIMSOPT seed artifact, DESC/VMEC equilibrium seed, and run output root from explicit CLI arguments.
   - [x] Emit a preflight JSON before optimization with resolved paths, source checksums, coil group counts, current signs, NFP, symmetry, hardware source metadata, and objective stack.
   - [x] Add an explicit equilibrium-load lane that writes `desc_equilibrium_load_report.json` without claiming DESC optimization.
   - [x] Add an explicit objective-assembly lane that writes `desc_runtime_coilset_build_report.json` and `desc_objective_assembly_report.json` without claiming DESC optimization.
   - [x] Add an explicit conversion-only Lane A smoke path that writes loadable bridge artifacts without claiming DESC optimization.
   - [x] Add an explicit fixed-equilibrium polish lane that defaults to a failed solve report before DESC optimizer execution, requires `--allow-high-memory-desc-optimizer` to call DESC `Optimizer`, writes `desc_fixed_polish_solve_report.json`, saves `desc_coils.h5` only after an opted-in successful solve, and keeps non-DESC validation gates blocked.
   - [x] Run fixed-equilibrium polish as the first executable stage and require it to pass round-trip validation before joint mode can be promoted.
   - [x] Run true joint mode with DESC `BoundaryError` or `VacuumBoundaryError`, never with `QuadraticFlux`.
   - [x] For conversion-only smoke runs, write `desc_result.json`, a sampled DESC-coil JSON artifact, exported SIMSOPT coil artifact, conversion reports, and validation manifest.
   - [x] For fixed-equilibrium DESC optimization runs, default to a failed `desc_result.json` plus solve report before runtime setup unless explicitly opted in; with explicit high-memory opt-in, pre-write the same failed result contract immediately before optimizer execution, then on a successful solve write `desc_coils.h5`, exported SIMSOPT coil artifact, conversion/import report, and validation manifest.
   - [x] For true joint DESC optimization runs, default to a failed `desc_result.json` plus solve report before runtime setup unless explicitly opted in; with explicit high-memory opt-in, pre-write the same failed result contract immediately before optimizer execution, then on a successful solve write DESC equilibrium/coil artifacts, exported SIMSOPT coil artifact, conversion/import report, and validation manifest.

9. Integrate SIMSOPT validation and final oracle
   - [x] Add a validation wrapper that accepts the exported DESC-optimized artifact plus existing SIMSOPT Poincare/Boozer sidecars, validates live artifact checksums, and writes physics evidence into the DESC joint validation manifest.
   - [x] Add orchestration that launches the high-cost SIMSOPT Poincare/Boozer validation runners for a DESC export instead of consuming precomputed sidecars.
   - [x] Reuse existing result fields where possible: `HARDWARE_CONSTRAINTS_OK`, `HARDWARE_CONSTRAINT_VIOLATIONS`, `BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK`, and final feasibility fields.
   - [x] Keep `search_hardware_status` and `artifact_hardware_status` separate in DESC joint run payloads.
   - [x] Add a final oracle gate that requires direct loaded-artifact hardware/contact evidence before setting promotion status to pass.
   - [x] Add orchestration that launches the existing direct hardware/contact oracle and writes checksum-bound `desc_joint_final_oracle_evidence_v1` before promotion can pass.
   - [x] Add a report generator that lists DESC objective success, SIMSOPT physics checks, final hardware oracle status, and artifact paths separately.

10. Add tests and regression fixtures
   - [x] Add focused SIMSOPT tests under `tests/geo/test_desc_bridge_conversion.py` for coil conversion, current sign preservation, group preservation, and round-trip metadata.
   - [x] Add focused SIMSOPT tests under `tests/geo/test_desc_joint_runner_contracts.py` for CLI preflight, mode selection, failure policy, and result schema.
   - [x] Add a synthetic conversion-only runner fixture proving the bridge writes a SIMSOPT-loadable exported `BiotSavart` and blocks promotion until validation/oracle evidence exists.
   - [x] Add a synthetic joint-run fixture proving the runner optimizes equilibrium and coils together, writes DESC artifacts plus a SIMSOPT-loadable exported `BiotSavart`, and blocks promotion until validation/oracle evidence exists.
   - [x] Add synthetic interrupt fixtures proving opted-in fixed-polish and
     joint optimizer lanes pre-write failed `desc_result.json` plus solve-report
     artifacts before entering DESC `Optimizer.optimize`.
   - [x] Add DESC-side unit tests for any new generic hardware objective before using it in the joint runner. Source tests for `CoilSetSDFDistance` exist in the paired DESC checkout and passed on the Perlmutter allocation in job `55112508`.
   - [x] Add a small synthetic integration fixture with one simple equilibrium and a small coil set to prove the end-to-end fixed-equilibrium polish path.
   - [x] Add a gated, non-default HBT smoke fixture for the first low-resolution vacuum joint solve. Perlmutter job `55107392` proves the `iota011_R0976` low-resolution `vacuum_joint` solve/export path with `desc_joint_runtime_solve_report.json` status `passed`; job `55108116` records the expected strict SIMSOPT physics-validation failure evidence for that moved-boundary artifact.
   - [x] Add regression tests that prevent `QuadraticFlux` from appearing in joint mode objective stacks.

11. Add execution lanes
   - [x] Lane A: fixed-equilibrium DESC coil polish from a known SIMSOPT banana artifact. The `winding_ext0009_R0p94_a0p15_FULLPASS` real seed now passes preflight, equilibrium load, objective assembly, sequential objective-value evaluation, sequential per-term scalar-gradient evaluation, resource-managed fixed-polish optimization, DESC `desc_coils.h5` save/reload, and SIMSOPT `BiotSavart` export/load. The current runner scopes optimizer variables to banana coils only (`dim_x: 340`) while preserving the full 30-coil field/objective state. Perlmutter GPU debug confirms CUDA startup is fixed via early DESC device bootstrap, and the formerly memory-heavy `CoilSetMinDistance` scalar-gradient path is now bounded: objective-gradient job `55103718` stayed flat at `16869808` ru_maxrss units through every term with Slurm max RSS `16744128K`. Fixed-polish optimizer smoke job `55104085` also stayed bounded with Slurm max RSS `16768408K` and failed only because the deliberate `--desc-maxiter 1` probe returned DESC `success=False`. Maxiter-10 job `55105011` completed `COMPLETED 0:0`, passed DESC optimizer solve/export, wrote `desc_coils.h5` and `biot_savart_desc_export.json`, and the exported artifact loads locally as a 30-coil SIMSOPT `BiotSavart`. Follow-up gallery sweep job `55105529` repeated the same fixed-polish/export proof for `iota011_R0976`, `iota011_R0935`, and `winding_band0015_R0p92_a0p15`, all with `optimizer_success: true`, loadable 30-coil SIMSOPT exports, and per-step Slurm MaxRSS below `16.6` GB. Physics validation, final hardware oracle, and promotion are separate downstream gates and remain fail-closed.
   - [x] Lane B runner path: vacuum joint DESC solve with banana coil geometry and plasma boundary moving together, behind explicit `--joint-run-only`. Perlmutter job `55107392` proves the low-resolution joint solve/export path for `iota011_R0976`; job `55108116` proves that stale moved-boundary artifact fails strict SIMSOPT physics validation cleanly (`0/50`, `surface_exit: 50`) with checksum-bound failed Boozer evidence. After the expanded-field runtime coilset fix, no patched Lane B export has a strict SIMSOPT validation pass yet.
   - [x] Lane B patched-candidate rerun after the expanded-field runtime
     coilset fix for `winding_band0015_R0p92_a0p15`. Perlmutter GPU-debug job
     `55148944` proves objective assembly and DESC/SIMSOPT field parity for the
     patched runtime coilset, and CPU-debug fallback job `55150212` completed
     the optimizer/export/strict-validation path. DESC solve/export passed, but
     strict SIMSOPT validation still failed with `0/50` surviving field lines,
     so this corrected seed does not satisfy the Lane C predecessor gate.
   - [x] Lane B patched selected-candidate sweep after the expanded-field
     runtime coilset fix. CPU-debug job `55150551` completed `COMPLETED 0:0`
     for `iota011_R0935` and `iota011_R0976`. Both DESC optimizer/export paths
     passed, but both strict SIMSOPT validations failed with `0/50` surviving
     field lines, so neither corrected selected seed satisfies the Lane C
     predecessor gate. A geometry audit of those exports showed the moved LCFS
     inflated from the seed volume band near `0.05` m^3 to roughly `0.21` m^3,
     so the active Lane B blocker is a joint-objective formulation issue rather
     than RAM or missing export plumbing.
   - [ ] Rerun Lane B after the joint `Volume` anchor and validation-surface
     binding fixes are synced to Perlmutter; require the exported LCFS volume to
     stay near the seed target and keep strict SIMSOPT validation separate from
     DESC optimizer success. Current CPU-debug evidence after syncing the hard
     `Volume` constraint shows the runner plumbing is correct but no
     promotion-ready Lane B artifact exists yet: job `55155873` proved
     `scipy-l-bfgs-b` is incompatible with nonlinear hard constraints, so the
     runtime now rejects such methods before `Optimizer.optimize`; job
     `55156136` showed `lsq-auglag` accepts the hard `Volume` constraint and
     keeps `iota011_R0935` / `iota011_R0976` at the seed volume band with
     bounded RSS (`~11`-`13` GB) but returns `optimizer_success: false`; job
     `55157483` showed `scipy-SLSQP` is not memory-safe for this seed under the
     64 GB CPU-debug cgroup; job `55166948` showed `fmin-auglag-bfgs` is
     memory-bounded (`5494788K` MaxRSS) and strongly reduces the objective
     (`9.310e6 -> 2.515e2`) but fails `success=True` and violates the hard
     volume target (`-4.920e-02 -> -1.104e+00` m^3), so export and strict
     validation correctly stay skipped. Job `55167099` showed
     `scipy-trust-constr` is also not viable for this seed/resolution under the
     CPU-debug cgroup: it reached objective assembly with the hard `Volume`
     constraint, then the solve step was Slurm-killed for OOM before any solve
     report or result existed (`OUT_OF_MEMORY|0:125`, `5253600K` step MaxRSS).
     Follow-up Perlmutter interactive CPU validation on the staged
     `iota011_R0935` gallery seed fixed the objective-eval RAM blowup and the
     DESC/SIMSOPT `G` convention mismatch without producing a promotion-ready
     Lane B artifact. The synced contract suite passed with `92 passed in
     56.44s`. Value-only job
     `runs/iota011_R0935_objective_eval_targetDescG_cpu_20260628T012117/`
     used the selected seed state sidecar, converted SIMSOPT `G = mu0*I` to
     DESC's `G = mu0*I/(2*pi)` convention, scaled `Psi` from `1.0` to
     `0.002313315932123337`, and reduced `LinkingCurrentConsistency` to
     `1.668928469551307e-06` scaled L2 with MaxRSS `2982084K`. The bounded
     optimizer rerun
     `runs/iota011_R0935_vacuum_joint_targetDescG_lsqauglag_cpu_20260628T012309/`
     stayed memory-bounded (MaxRSS `12442372K`) and kept linking-current error
     near zero (`2.937 -> 7.331` A), but still returned
     `optimizer_success: false` after `--desc-maxiter 10`; no DESC export or
     strict SIMSOPT validation artifact was produced. Follow-up optimizer-control
     hardening added typed `lsq-auglag` controls (`ftol`, `xtol`, `gtol`,
     `ctol`, `max_nfev`, and `min_trust_radius`) and rejects proximal wrappers
     when non-equilibrium hard constraints remain in the joint stack. Perlmutter
     CPU-debug jobs `55174344` and `55174419` validated that slice with
     `4 passed, 97 deselected` and `101 passed`, respectively, under `--mem=8G`.
     A bounded continuation probe (`55175281`) stayed below the 32 GB cgroup but
     exposed a real formulation mismatch: a `vacuum_joint` run was loading a DESC
     H5 seed with nonzero pressure and toroidal-current profiles, so DESC warned
     that `VacuumBoundaryError` was incorrect. The runner now normalizes loaded
     DESC equilibrium profiles only for `vacuum_joint`, records
     `mode_profile_adjustment` in `desc_equilibrium_load_report.json`, and leaves
     non-vacuum modes unchanged. Focused contract job `55175840` passed
     `4 passed, 97 deselected`; full contract job `55176329` passed
     `101 passed in 70.77s` with batch MaxRSS `779928K`. The real patched
     continuation run (`55175895`) removed the nonzero-pressure/current
     `VacuumBoundaryError` warnings and converted the previous no-step failure
     (`nit=1`, `nfev=121`, `njev=0`, `Total delta_x=0.0`) into an actual bounded
     optimizer trajectory (`nit=20`, `nfev=48`, `Jacobian evaluations=20`,
     `Total delta_x=2.971e-01`, batch MaxRSS `17088640K`). It still returned
     `success=false` with `Maximum number of iterations has been exceeded`, wrote
     only failed DESC checkpoints, and produced no exported SIMSOPT artifacts.
     A diagnostic `finite_beta_joint` probe from the same non-vacuum checkpoint
     (`55176508`) preserved pressure/current as intended but was OOM-killed at
     optimizer startup under `--mem=32G` after `/usr/bin/time` recorded
     `33367716K` max RSS, so finite-beta is not the right continuation from this
     checkpoint/debug envelope. The clean fresh-seed Lane B probe,
     `iota011_R0935` from the original SIMSOPT field/LCFS seed with no failed
     DESC coil checkpoint, reached equilibrium load, runtime coilset build,
     objective assembly, and optimizer-scope reporting under job `55177787`.
     It did not OOM (`18988416K` batch MaxRSS under `--mem=32G`), but hit the
     30-minute debug walltime inside the optimizer before writing optimized
     artifacts. Because that run predated the pre-optimizer result-contract fix,
     it exposed the missing `desc_result.json` / solve-report artifact on
     scheduler kill. The patched rerun, job `55179895`, reached the optimizer,
     wrote the pre-optimizer fail-closed `desc_result.json` plus
     `desc_joint_runtime_solve_report.json`, and preserved those artifacts after
     scheduler cancellation at `00:04:01` with batch MaxRSS `10187148K` under
     `--mem=32G`. A further hard-constraint continuation from the latest failed
     checkpoints, CPU-debug job `55180407`, also stayed memory-bounded
     (`19824744K` batch MaxRSS under `--mem=32G`) and preserved the
     pre-optimizer failed/blocked result contract, but again returned
     `optimizer_success: false` after `nit=20`, `nfev=49` with `Maximum number
     of iterations has been exceeded`. The DESC summary showed no improvement
     in the active metrics (`Total (sum of squares): 1.444e-04 -> 1.444e-04`,
     `Constraint violation: 1.646e-02`, unchanged volume and force residuals)
     despite `Total delta_x=5.798e-01`, and wrote only failed DESC checkpoints.
     Repeated checkpoint-only continuation is therefore not closing Lane B. The
     remaining blocker is optimizer/formulation progress against the
     boundary/force/curvature/distance stack, not field-current scaling, profile
     hygiene, contract coverage, export plumbing, failure-artifact durability,
     or raw memory. The next explicit staging path is
     `--desc-joint-constraint-policy proximal-force-balance`, which leaves
     `ForceBalance` projected as the hard DESC equilibrium constraint and stages
     `Volume` as a weighted objective because DESC `ProximalProjection` rejects
     general nonlinear hard constraints. That staged policy is now source-level
     and contract-test validated: CPU-debug job `55182061` passed py-compile plus
     the nine focused hard/proximal assembly, runner, and runtime-solve contract
     tests under explicit `--mem=8G`. The first real-seed staged probe,
     CPU-debug job `55182186`, assembled the intended stack
     (`objective_names` include `Volume`, `constraint_names == ['ForceBalance']`,
     `joint_constraint_policy: proximal-force-balance`) and entered DESC
     `proximal-lsq-exact` with `674` parameters and `4444` objectives. It timed
     out at the 30-minute debug walltime before DESC returned optimizer results,
     so it produced no optimized artifacts, but Slurm batch MaxRSS stayed bounded
     at `9012672K` under `--mem=32G`. The proximal projection was active: stderr
     showed the DESC current-on-axis warning decreasing from about `-1.558e+02`
     A to `1.253e-03` A before timeout. A follow-up capped-proximal probe,
     CPU-debug job `55184016`, used `--desc-proximal-perturb-order 1`,
     `--desc-proximal-solve-maxiter 2`, and
     `--no-desc-proximal-solve-during-build`. It returned before walltime with
     `optimizer_nit: 1`, `optimizer_nfev: 30`, DESC message
     `A bad approximation caused failure to predict improvement.`, and failed
     optimizer checkpoints, while remaining memory-bounded (`8514952K` max RSS
     by `/usr/bin/time`, `9149048K` Slurm batch MaxRSS under `--mem=32G`). This
     converts the staged-proximal blocker from RAM/timeout to optimizer progress:
     no promoted DESC/SIMSOPT artifacts exist yet. Patched vacuum Lane B is
     memory-bounded so far, but finite-beta remains gated by both the missing
     Lane B predecessor and the larger constrained solve footprint.
   - [ ] Lane C: finite-beta joint DESC solve after Lane B has a passing validation artifact. Current source gates are ready and fail closed on missing/failed Lane B evidence, but Perlmutter jobs `55121657` and `55150551` confirmed the currently staged extra Lane B seeds still do not provide a passing prerequisite; diagnostic job `55176508` also showed that continuing `finite_beta_joint` from the non-vacuum failed checkpoint is not memory-safe under the 32 GB CPU-debug cgroup.
   - [x] Lane D: hardware-space exploration using the same DESC joint runner but multiple hardware spec files. Perlmutter job `55120742` used the same `iota011_R0976` `vacuum_joint` runner and evaluated both the base point-cloud keepout spec and the corrected SDF-only spec in value-only mode; both assembled and evaluated finite objective reports, while the SDF spec added `HardwareSdfKeepout` / `CoilSetSDFDistance` search steering.
   - [x] Record lane outputs in an inventory file with absolute artifact paths, source hashes, run mode, objective stack, validation status, and promotion status.

12. Performance and memory hardening
   - [x] Add low-resolution defaults for local smoke tests and explicit production-resolution configs for real runs.
   - [x] Record Biot-Savart chunk sizes, DESC source/eval grids, coil Fourier order, sample counts, and memory-relevant settings in every run payload.
   - [x] Add timing sections for conversion, objective build, first objective evaluation, first gradient evaluation, optimizer iterations, export, and validation.
   - [x] Add a benchmark smoke command that runs one objective and gradient evaluation without launching a long optimization.
   - [x] Keep high-cost JIT/Jacobian/gradient work behind explicit runner flags or production configs, not implicit test defaults.
   - [x] Fail closed before DESC fixed-polish/joint optimizer execution unless `--allow-high-memory-desc-optimizer` is supplied; real-seed default-block evidence lives at `runs/winding_ext0009_fixed_polish_default_blocked_20260626T181351Z/`.
   - [x] Add early DESC runtime device bootstrap so GPU runs select DESC's device before any `desc.backend` import.
   - [x] Reduce Lane A optimizer variables to the banana coil group before invoking DESC `Optimizer`; current real-seed evidence shows 10 optimized banana coils, 20 fixed TF/proxy/VF coils, and value/gradient-smoke `dim_x: 340`.
   - [x] Route blocked scalar-gradient diagnostics through DESC `compute_quadratic_scalar` and add a memory-bounded hard-min scalar-gradient path for DESC `CoilSetMinDistance`; Perlmutter job `55103718` proves the real-seed per-term gradient path no longer climbs from ~16.8 GB to ~43.4 GB.
   - [x] Reject joint-mode DESC optimizer methods that do not support equality
     constraints when hard constraints are assembled. This prevents DESC from
     silently falling into incompatible nonlinear-constraint wrapper paths after
     the `Volume` anchor is promoted to a hard constraint. Perlmutter
     CPU-debug job `55167007` passed the synced contract suite with
     `89 passed in 63.64s` and Slurm batch MaxRSS `774880K`.

13. Documentation and handoff
   - [x] Update `docs/desc_banana_single_stage_feasibility_report_2026-06-26.md` after the first implementation slice lands.
   - [x] Add `examples/single_stage_optimization/DESC_JOINT/README.md` with a minimal command sequence for Lane A and Lane B.
   - [x] Add a troubleshooting section for current sign mismatch, NFP/symmetry mismatch, LCFS seed mismatch, failed hardware provenance, and failed oracle status.
   - [x] Add a result interpretation guide that clearly separates DESC solve success from physics validation success and final hardware promotion.

## Validation Plan

Current-checkout note: the SIMSOPT-side contract tests now exist and are part
of the active validation slice. Use the repo `.conda-env` for SIMSOPT-side
contract tests. Use `/opt/homebrew/Caskroom/miniforge/base/bin/python3` for
local DESC runtime smoke and DESC-side tests, because that interpreter imports
the paired DESC checkout plus SIMSOPT, JAX, `h5py`, `termcolor`, and
`colorama`.

- [x] `git -C /Users/suhjungdae/code/columbia/simsopt-surrogate diff --check`
- [x] `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_desc_bridge_conversion.py -q`
- [x] `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && PYTHONNOUSERSITE=1 ./.conda-env/bin/python -m pytest tests/geo/test_desc_joint_runner_contracts.py -q`
- [x] `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 ./.conda-env/bin/python -m pytest -q -p no:cacheprovider tests/geo/test_hardware_keepout_winding_frame.py tests/geo/test_stage2_formulation_audit_fixes.py`
- [x] `cd /Users/suhjungdae/code/opensource/DESC && PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_objective_funs.py -k "BoundaryError or VacuumBoundaryError or QuadraticFlux or LinkingCurrentConsistency" -q`
- [x] `cd /Users/suhjungdae/code/opensource/DESC && PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/Caskroom/miniforge/base/bin/python3 -m pytest tests/test_optimizer.py -k "optimize_coil_currents" -q`
- [x] Run SIMSOPT -> DESC -> SIMSOPT round-trip on a synthetic fixture and compare coordinate, current, length, distance, and field-sample residuals.
- [x] Run real-seed preflight for `.desc-joint-real-seed-smoke/` using `winding_ext0009_R0p94_a0p15_FULLPASS`.
- [x] Run real-seed DESC equilibrium load for `winding_ext0009_R0p94_a0p15_FULLPASS`; LCFS parity max XYZ delta is `3.8011967608597686e-15` m over 441 samples.
- [x] Run generated NFP=5 DESC H5 seed load for `winding_ext0009_R0p94_a0p15_FULLPASS`; LCFS parity max XYZ delta is `0.0` m over 441 samples.
- [x] Run generated NFP=5 VMEC wout seed load for `winding_ext0009_R0p94_a0p15_FULLPASS`; VMEC-boundary LCFS parity max XYZ delta is `7.906990629795298e-16` m over 441 samples.
- [x] Run real-seed DESC objective assembly for `winding_ext0009_R0p94_a0p15_FULLPASS`; it writes `desc_runtime_coilset_build_report.json` and `desc_objective_assembly_report.json`.
- [x] Run real-seed DESC objective value evaluation for `winding_ext0009_R0p94_a0p15_FULLPASS`; sequential per-term smoke writes `runs/winding_ext0009_objective_eval_sequential/desc_objective_evaluation_report.json` with all six terms finite.
- [x] Run real-seed DESC combined Jacobian evaluation for `winding_ext0009_R0p94_a0p15_FULLPASS`; Perlmutter CPU-debug job `55121055` completed `COMPLETED 0:0`, wrote `winding_ext0009_combined_jacobian_summary_55121055.json`, reported `dim_x=340`, `dim_f=3762`, Jacobian shape `[3762, 340]`, `scaled_error_all_finite: true`, `jacobian_all_finite: true`, `jacobian_seconds: 269.3061958710896`, and Slurm step max RSS `21802760K`. This remains explicit opt-in and banana-scope only; the runner still keeps optimizer execution behind `--allow-high-memory-desc-optimizer`.
- [x] Run real-seed fixed-polish default guard for `winding_ext0009_R0p94_a0p15_FULLPASS`; without `--allow-high-memory-desc-optimizer` it writes a failed `desc_result.json` before runtime coilset/objective construction and records peak footprint `365151600` bytes.
- [x] Run Perlmutter GPU-debug fixed-polish probe for `winding_ext0009_R0p94_a0p15_FULLPASS`; jobs `55098630`, `55100246`, and `55100694` bootstrap DESC on GPU, pass equilibrium load/coilset build/objective assembly, then fail the opted-in `scipy-l-bfgs-b` optimizer path with status `99`, no export, and about `43`-`44` GB max RSS. Source audit showed `lsq-exact` is a dense-Jacobian path, so job `55101083` was canceled rather than used as the low-memory probe. Jobs `55102191` and `55102355` ran the explicit sequential per-term gradient diagnostic; job `55102355` passed and attributed the RSS jump to `CoilSetMinDistance` gradient (`16845060` ru_maxrss units before the term, `43450888` after it; Slurm max RSS `43369756K`). Job `55103434` proved the first DESC-only hook was bypassed by the active smoke evaluator and still hit `43363780K`. Job `55103718` then proved the active evaluator plus DESC hard-min scalar-gradient patch fixed the spike: all terms stayed at `16869808` ru_maxrss units and Slurm max RSS was `16744128K`. Job `55104085` ran an opted-in `scipy-l-bfgs-b` fixed-polish smoke with `--desc-maxiter 1`; it stayed bounded at Slurm max RSS `16768408K`, passed objective assembly, wrote solve/result reports, and failed only because the deliberate one-iteration optimizer smoke returned `success=False`.
- [x] Run Lane A fixed-equilibrium polish and confirm exported artifact can be loaded by existing SIMSOPT validation; Perlmutter job `55105011` completed `COMPLETED 0:0`, passed fixed-polish solve/export with Slurm max RSS `16810932K`, wrote `desc_coils.h5` plus `biot_savart_desc_export.json`, and the pulled export loads locally as a `BiotSavart` with 30 coils.
- [x] Run selected 2026-06-25 gallery seeds through the same Lane A Perlmutter debug-GPU polish/export lane; job `55105529` passed for `iota011_R0976`, `iota011_R0935`, and `winding_band0015_R0p92_a0p15`, and each pulled export loads locally as a 30-coil SIMSOPT `BiotSavart`.
- [x] Run post-export SIMSOPT physics validation on the selected Lane A exports; all three Boozer re-solves passed, but strict validation-mode Poincare failed at `42/50` for each, so promotion remains fail-closed.
- [x] Run final hardware/contact oracle on at least one DESC-exported candidate before marking promotion-ready; `iota011_R0976` passed the direct contact audit with `0/10` contacts and wrote checksum-bound final-oracle evidence, but promotion remains failed because physics validation did not pass.
- [x] Run Lane B low-resolution vacuum joint solve and confirm result payload separates DESC success, physics validation, and hardware oracle status. Job `55107392` produced a successful DESC joint solve/export; job `55108116` produced the strict validation manifest with physics `passed: false`, promotion `failed`, Poincare `0/50`, and a failed Boozer sidecar bound to `surf_desc_equilibrium_export.json`.
- [x] Run real-seed runtime coilset field-parity validation after the
  expanded-field `NFP=1, sym=False` fix. Perlmutter GPU-debug job `55148944`
  completed `COMPLETED 0:0`, wrote
  `winding_band0015_R0p92_a0p15_fieldparity_summary_55148944.json`, passed
  objective assembly, and recorded `max_delta_T: 0.016951885063729997` /
  `mean_delta_T: 0.005167423993795285` instead of the stale prerequisite-search
  run's `3`-`4` T DESC/SIMSOPT field mismatch.
- [x] Rerun the Lane B optimizer and strict SIMSOPT validation on the patched
  expanded-field runtime coilset for `winding_band0015_R0p92_a0p15`.
  CPU-debug job `55150212` completed `COMPLETED 0:0`, wrote
  `winding_band0015_R0p92_a0p15_vacuum_joint_patched_cpu_summary_55150212.json`,
  passed DESC solve/export with `optimizer_success: true`, `optimizer_nit: 3`,
  and Slurm step max RSS `5215324K`, but strict validation still failed with
  `0/50` surviving field lines and all `surface_exit`; no Lane C predecessor
  claim is proven for this seed.
- [x] Rerun the remaining selected Lane B candidates on the patched expanded
  field runtime coilset. CPU-debug job `55150551` completed `COMPLETED 0:0`,
  wrote `lane_b_patched_selected_sweep_summary_55150551.json`, and solved both
  `iota011_R0935` and `iota011_R0976` with `optimizer_success: true`. The
  strict SIMSOPT validations failed for both exports with `0/50` surviving
  field lines, so no patched selected Lane B seed proves the Lane C predecessor
  claim. Geometry audit of the same outputs showed the moved LCFS inflated from
  the seed volume band near `0.05` m^3 to roughly `0.21` m^3, so the current
  root fix is the joint `Volume` objective anchor plus a rerun, not additional
  local memory tuning.
- [x] Run real-seed SDF hardware-keepout objective assembly/value evaluation for `iota011_R0976`; Perlmutter CPU-debug fallback job `55120328` completed `COMPLETED 0:0`, wrote `desc_objective_assembly_report.json` with `HardwareSdfKeepout` and manifest-bound SDF provenance, wrote `desc_objective_evaluation_report.json` with `evaluation_mode: sequential_terms`, final term `CoilSetSDFDistance`, `dim_x: 2891`, `dim_f: 4623`, `scaled_error_all_finite: true`, no Jacobian/gradient, and Slurm step max RSS `4905412K`. GPU-debug job `55119473` was canceled before allocation after earlier GPU allocation startup had already been proven by `55112508`/`55114058`.
- [x] Run Lane D multi-spec hardware exploration through the same DESC joint runner; Perlmutter CPU-debug job `55120742` completed `COMPLETED 0:0`, wrote `lane_d_multispec_summary_55120742.json`, and produced finite sequential value-only objective reports for both `.desc-joint-real-seed-smoke/hardware_spec.json` (`dim_f=4323`, six objective terms, Slurm step max RSS `4157660K`) and `.desc-joint-real-seed-smoke/hardware_spec_with_sdf_only_perlmutter.json` (`dim_f=4623`, additional `HardwareSdfKeepout` / `CoilSetSDFDistance`, Slurm step max RSS `4115408K`). The job did not compute Jacobians, gradients, or run the DESC optimizer.
- [x] Run finite-beta promotion-gate regression coverage on Perlmutter after adding `lane_b_predecessor_status`; the synced scratch checkout first passed the three focused Lane B predecessor tests with `3 passed in 5.09s`, then passed the full `tests/geo/test_desc_joint_runner_contracts.py` suite. The latest synced Perlmutter contract suite after the expanded-field runtime coilset fix passed with `84 passed in 97.27s`.
- [x] Run the synced SIMSOPT-side contract suite on Perlmutter after changing
  joint `Volume` from a soft objective to a hard constraint and adding the
  optimizer capability guard. CPU-debug job `55167007` completed
  `COMPLETED 0:0` with `89 passed in 63.64s`; Slurm batch MaxRSS was
  `774880K`.
- [x] Finish the remaining hard-`Volume` Lane B optimizer-method probe for
  `scipy-trust-constr`; CPU-debug job `55167099` reached objective assembly
  with the hard `Volume` constraint and then failed in the solve step with a
  Slurm OOM kill (`OUT_OF_MEMORY|0:125`, step MaxRSS `5253600K`), before
  writing a solve report or result.
- [x] Rerun the focused DESC-joint contract suite and full
  `tests/geo/test_desc_joint_runner_contracts.py` suite on Perlmutter after
  adding SIMSOPT-sidecar-to-DESC-`G` conversion; the full suite passed with
  `92 passed in 56.44s` under explicit `--mem=8G`.
- [x] Rerun `iota011_R0935` real-seed sequential objective-value evaluation
  after the `G/(2*pi)` convention fix; Perlmutter output
  `runs/iota011_R0935_objective_eval_targetDescG_cpu_20260628T012117/` passed,
  reported `target_lcfs_G: -0.3199994125225332`, `Psi:
  0.002313315932123337`, all terms finite, `LinkingCurrentConsistency` scaled
  L2 `1.668928469551307e-06`, and MaxRSS `2982084K`.
- [x] Rerun bounded `iota011_R0935` Lane B `lsq-auglag --desc-maxiter 10`
  after the `G/(2*pi)` convention fix; Perlmutter output
  `runs/iota011_R0935_vacuum_joint_targetDescG_lsqauglag_cpu_20260628T012309/`
  stayed bounded at MaxRSS `12442372K` and kept linking-current error near zero
  (`2.937 -> 7.331` A), but returned `success=False` at maxiter with no exported
  artifacts.
- [x] Validate typed optimizer controls, proximal-wrapper rejection when hard
  joint constraints include non-equilibrium terms, and vacuum-profile preparation
  on Perlmutter. Focused CPU-debug jobs `55174344` and `55175840` each passed `4 passed,
  97 deselected`; full-suite CPU-debug jobs `55174419` and `55176329` each
  passed `101` contract tests under explicit `--mem=8G`.
- [x] Validate the shared profile-preparation report contract for load-only,
  `vacuum_joint`, and `finite_beta_joint` lanes after the non-vacuum preservation
  branch was made observable. CPU-debug job `55178421` passed the three focused
  profile-mode contract tests in `13.17s` under explicit `--mem=8G`.
- [x] Validate pre-optimizer fail-closed result materialization for both opted-in
  optimizer lanes on Perlmutter. CPU-debug job `55179599` ran `py_compile` on
  `DESC_JOINT/run_desc_joint_banana.py` and passed seven focused contract tests,
  including the fixed-polish and joint `fixture-interrupt` regressions, in
  `23.28s` under explicit `--mem=8G`.
- [x] Rerun bounded `iota011_R0935` Lane B continuation after normalizing loaded
  profiles for `vacuum_joint`; job `55175895` stayed below the 32 GB cgroup
  (batch MaxRSS `17088640K`), removed the incorrect nonzero-pressure/current
  `VacuumBoundaryError` warnings, and performed real `lsq-auglag` iterations
  (`nit=20`, `nfev=48`, `Jacobian evaluations=20`, `Total delta_x=2.971e-01`),
  but still returned `success=False` at maxiter with no exported artifacts.
- [x] Probe `finite_beta_joint` on the same non-vacuum failed checkpoint; job
  `55176508` preserved pressure/current as intended but was OOM-killed at
  optimizer startup under `--mem=32G` after `/usr/bin/time` recorded
  `33367716K` max RSS, so this is not a viable continuation path in the current
  CPU-debug envelope.
- [x] Rerun bounded `iota011_R0935` Lane B `vacuum_joint` from the fresh staged
  SIMSOPT-field/LCFS seed, without a failed DESC coil checkpoint. Perlmutter job
  `55177787` reached equilibrium load, runtime coilset build, objective
  assembly, and optimizer-scope reporting, then timed out inside the optimizer
  after `00:30:20`; Slurm recorded batch MaxRSS `18988416K` under explicit
  `--mem=32G`, so this probe was timeout-limited rather than memory-killed. It
  produced setup reports but no optimized artifacts. Because it predated the
  pre-optimizer result-contract fix, it also produced no `desc_result.json` or
  solve report after the scheduler kill.
- [x] Rerun the same fresh-seed Lane B probe with the pre-optimizer result
  contract fix synced. Perlmutter job `55179895` reached the optimizer, wrote
  fail-closed `desc_result.json` and `desc_joint_runtime_solve_report.json`,
  then preserved both after scheduler cancellation at `00:04:01`; Slurm
  recorded batch MaxRSS `10187148K` under explicit `--mem=32G`.
- [x] Rerun hard `Volume` + `ForceBalance` Lane B continuation from the latest
  failed DESC checkpoints. CPU-debug job `55180407` stayed memory-bounded
  (`19824744K` batch MaxRSS under explicit `--mem=32G`) and wrote failed
  checkpoint artifacts, but returned `optimizer_success: false` at maxiter
  (`nit=20`, `nfev=49`) with unchanged objective, constraint violation, volume,
  and force residuals. This rules out another blind checkpoint-only continuation
  as the next credible Lane B closure step.
- [ ] Validate the explicit staged proximal Lane B policy on Perlmutter:
  `--desc-joint-constraint-policy proximal-force-balance` with DESC
  `proximal-lsq-exact`. This path must keep `ForceBalance` as the only hard
  projected equilibrium constraint, record `Volume` as an objective in
  `desc_objective_assembly_report.json`, preserve CW/current-sign convention
  evidence, and remain bounded under explicit Slurm `--mem`.
- [x] Validate staged proximal policy source contracts on Perlmutter. CPU-debug
  job `55182061` ran py-compile for the touched DESC joint runner/factory/solve
  modules and passed the nine focused hard/proximal assembly, CLI, and
  runtime-solve tests in `12.68s` under explicit `--mem=8G`.
- [x] Validate typed proximal-control contracts on Perlmutter. CPU-debug job
  `55183856` ran py-compile for the touched runner/solve/test modules and passed
  the five focused proximal-control tests in `7.15s` under explicit `--mem=8G`.
- [x] Record the CW/current-sign convention explicitly in DESC bridge reports.
  Export, import, and static BiotSavart inventory payloads now include
  `coil_conventions` with source-parameterization preservation,
  signed-current preservation, and HBT negative-current TF CW convention.
  CPU-debug job `55184830` ran py-compile for the touched convention/report
  modules and passed the four focused convention/inventory tests in `2.59s`
  under explicit `--mem=8G`; after fixing stale fixed-polish expectations for
  the recorded default joint constraint policy, CPU-debug job `55228586` passed
  the full `test_desc_bridge_conversion.py` plus
  `test_desc_joint_runner_contracts.py` set with `124 passed in 80.49s` and
  Slurm batch MaxRSS `810320K` under explicit `--mem=8G`.
- [x] Run first real-seed staged proximal Lane B probe from the latest hard
  Volume+ForceBalance failed checkpoints. CPU-debug job `55182186` assembled
  `proximal-force-balance` correctly (`Volume` objective, hard `ForceBalance`
  only), entered DESC `proximal-lsq-exact`, and stayed memory-bounded
  (`9012672K` batch MaxRSS under explicit `--mem=32G`). It timed out at the
  30-minute debug walltime before DESC returned, so no optimized artifacts or
  failed optimizer checkpoints were produced beyond the pre-optimizer
  fail-closed result contract. This proves the previous proximal API blocker and
  RAM blowup are addressed for this path; the remaining Lane B blocker is
  proximal solve latency/progress to a returned optimizer result.
- [x] Rerun the staged proximal Lane B probe with explicit DESC proximal-wrapper
  caps: `--desc-proximal-perturb-order 1`,
  `--desc-proximal-solve-maxiter 2`, and
  `--no-desc-proximal-solve-during-build`. DESC upstream tests use the same
  option family to bound proximal solves, and job `55184016` showed the capped
  path returns before walltime (`15:24.50` elapsed, `optimizer_nit: 1`,
  `optimizer_nfev: 30`) with bounded memory (`8514952K` `/usr/bin/time` max RSS,
  `9149048K` Slurm batch MaxRSS under `--mem=32G`). DESC reported
  `A bad approximation caused failure to predict improvement.`, wrote failed
  optimizer checkpoints, and produced no optimized artifacts. The staged
  proximal path is therefore no longer blocked by RAM or default projection
  walltime; it is blocked by lack of optimizer progress from the current
  objective/formulation schedule.

## Risks and Mitigations

- Risk: SIMSOPT and DESC coordinate/sign conventions silently diverge.
  Mitigation: Require deterministic surface samples, coil samples, current signs,
  CW handedness/orientation, NFP, symmetry, and field samples in every conversion
  test and run report.

- Risk: Hardware steering is mistaken for final buildability.
  Mitigation: Preserve `search_hardware_status` versus `artifact_hardware_status` and require direct loaded-artifact oracle evidence for promotion.

- Risk: DESC `QuadraticFlux` is accidentally used in a joint equilibrium-plus-coil solve.
  Mitigation: Add an objective-factory test and runner preflight that reject `QuadraticFlux` in joint mode before optimizer construction.

- Risk: A native DESC banana coil class becomes HBT-specific and shallow.
  Mitigation: Start with `FourierXYZCoil.from_values`; only add a native class if conversion residuals or performance justify it, and keep generic API boundaries in DESC.

- Risk: The current large SIMSOPT script absorbs DESC-specific complexity.
  Mitigation: Add a new `DESC_JOINT` runner and `banana_opt/desc_bridge` package instead of expanding `single_stage_banana_example.py` first.

- Risk: Long joint solves fail late because seed artifacts are malformed.
  Mitigation: Add preflight checksum, group, current, NFP, symmetry, hardware-provenance, and conversion-residual gates before optimization.

- Risk: Differentiable hardware SDF in DESC duplicates SIMSOPT hardware policy.
  Mitigation: Keep DESC objective generic and preserve SIMSOPT/CAD as the final policy layer.

- Risk: Tests become mock-only wiring checks.
  Mitigation: Require observable state assertions: coordinates, currents, group labels, objective stack contents, result payload fields, and loadable exported artifacts.

## Completion Criteria

- [x] A hardware-first spec can be loaded and validated without reading Markdown or relying on environment variables.
- [x] A known SIMSOPT banana artifact can be converted to DESC coils and back with explicit residuals and passing tests.
- [x] A DESC equilibrium seed can be created from a DESC or VMEC/wout source with provenance and LCFS parity checks.
- [x] Lane A fixed-equilibrium polish runs end-to-end and exports a SIMSOPT-loadable artifact.
- [x] Lane B low-resolution vacuum joint solve runs end-to-end with
  `BoundaryError` or `VacuumBoundaryError`; stale and patched expanded-field
  candidates both fail strict SIMSOPT physics validation, and the later hard
  `Volume` constrained reruns have not yet produced a successful
  optimizer/export/strict-validation artifact. Lane C therefore remains blocked
  on a passing Lane B predecessor, not on missing runner/export plumbing.
- [x] Result payloads separately report DESC solve status, SIMSOPT physics validation, search hardware status, artifact hardware status, and promotion status.
- [x] No joint-mode objective stack includes `QuadraticFlux`.
- [x] Joint-mode objective stacks can include manifest-bound in-loop hardware SDF keepout as a search-time steering term while keeping final promotion tied to SIMSOPT/CAD oracle evidence.
- [x] Final promotion requires existing SIMSOPT Poincare/Boozer validation plus direct hardware oracle evidence.
- [x] Documentation includes command examples, artifact paths, validation commands, and interpretation rules.

## Current Decisions and Open Questions

- Decision: Lane A remains anchored on `winding_ext0009_R0p94_a0p15_FULLPASS`
  for the first memory-bounded fixed-polish artifact, with the selected-gallery
  sweep (`iota011_R0976`, `iota011_R0935`, and
  `winding_band0015_R0p92_a0p15`) providing additional loadable export evidence.
- Decision: Lane B starts with `vacuum_joint`; `finite_beta_joint` remains Lane C
  and requires a passing Lane B vacuum-joint predecessor before promotion can
  pass.
- Decision: the canonical DESC-to-SIMSOPT export package is DESC HDF5
  (`desc_equilibrium.h5`, `desc_coils.h5`) plus conversion/import sidecars and a
  SIMSOPT-loadable JSON export. MAKEGRID remains optional downstream
  interoperability, not the promotion SSOT.
- Decision: final promotion SSOT is direct loaded-artifact SIMSOPT/CAD
  hardware/contact oracle evidence, not DESC search-time hardware steering.
- Open: define the production acceptance tolerance for magnetic-field sample
  parity near the plasma boundary. Synthetic conversion tests assert `1e-12 T`,
  while real runtime parity reports currently use the fail-closed `2.0 T` guard
  to catch broken field conventions rather than certify production accuracy.
- Open: decide when to expand the optimized coil groups beyond banana coils.
  The current low-memory lane fixes proxy/VF/TF coils and optimizes banana coils
  only; adding proxy or VF variables needs explicit bounds and separate memory
  evidence.
- Open: identify a Lane B optimizer/formulation schedule that produces an
  exported artifact passing strict SIMSOPT physics validation. Current hard
  `Volume`, profile-normalized, and staged-proximal runs prove the runner
  contracts and bounded memory for patched vacuum Lane B, but not feasibility.
