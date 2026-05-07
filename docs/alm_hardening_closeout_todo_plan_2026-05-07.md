# ALM Hardening Closeout Todo Plan

Date: 2026-05-07

Validated against: `simsopt-surrogate` HEAD `fd18380c6`

Status: open closeout tracker for the remaining review findings after `docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md` was implemented.

Relationship: this tracker closes the remaining review-update items from `docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md`; it does not replace the implemented phase history in that document.

## Goal

Close the remaining ALM/off-spec review gaps without reintroducing defensive fallbacks, hidden compatibility shims, or duplicate contract surfaces.

## Priority 0: Scope and Contract Alignment

- [ ] Decide the off-spec deletion path.
  - Option A: split the off-spec deletion into its own commit, preferably `refactor: remove off-spec engineering escape hatch`.
  - Option B: keep it bundled and amend `docs/alm_scalar_hardening_block_penalty_removal_plan_2026-05-06.md` so the commit plan and implementation note name the off-spec deletion as intentional scope.
  - Scope: `ACCEPT_OFFSPEC_*`, `accept_offspec_r0_seed`, `allow_offspec_engineering_constraints`, and related CLI/env escape paths.
  - Acceptance: the original ALM hardening work no longer silently carries an unrelated hardware-contract policy deletion.

- [ ] Update the conflicting project memory / operator guidance after the decision above.
  - Current conflicting memory: `/Users/suhjungdae/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/project_hbt_sidecar_conventions.md`.
  - This is a session-side memory action outside the git-tracked repo.
  - Acceptance: the memory either points to the tracked strict-rejection contract doc or explicitly documents the off-spec deprecation path.

## Priority 1: ALM CLI SSOT

- [ ] Replace the two independent ALM CLI tuples with one shared field table in `workflow_runner_common.py`.
  - Current parser tuple: `_SINGLE_STAGE_ALM_ARGUMENTS` in `run_80ka_baseline_tradeoff_sweep.py`.
  - Current emitter tuple: `_ALM_CLI_FLAG_ATTR_SUFFIXES` in `workflow_runner_common.py`.
  - Acceptance: parser registration and subprocess command emission derive from the same source.

```python
ALM_CLI_FIELDS = (
    ("max_outer_iters", int, 10),
    ("penalty_init", float, 1.0),
    ("penalty_scale", float, 10.0),
    ("penalty_max", float, 1.0e8),
    ("feas_tol", float, 1.0e-6),
    ("stationarity_tol", float, 1.0e-6),
    ("trust_radius_init", float, 0.05),
    ("trust_radius_min", float, 1.0e-4),
    ("trust_radius_shrink", float, 0.5),
    ("trust_radius_grow", float, 1.5),
    ("max_inner_attempts", int, 4),
    ("max_subproblem_continuations", int, 20),
    ("distance_smoothing", float, 0.005),
    ("curvature_smoothing", float, 0.05),
)

def alm_flag(suffix: str) -> str:
    return f"--alm-{suffix.replace('_', '-')}"

def single_stage_alm_flag(suffix: str) -> str:
    return f"--single-stage-alm-{suffix.replace('_', '-')}"
```

- [ ] Eliminate the old tuple names after migration.
  - Delete `_ALM_CLI_FLAG_ATTR_SUFFIXES` and `_SINGLE_STAGE_ALM_ARGUMENTS`.
  - Consumers should iterate `ALM_CLI_FIELDS` directly or call shared helpers derived from it.
  - Acceptance: `git grep -nE "_ALM_CLI_FLAG_ATTR_SUFFIXES|_SINGLE_STAGE_ALM_ARGUMENTS" examples/single_stage_optimization tests/geo` returns zero hits.

- [ ] Add a schema parity test.
  - Acceptance: `ALM_CLI_FIELDS`, parser-derived flags, and emitter-derived flags have the same suffix set and count.

- [ ] Add a round-trip parity test.
  - Flow: `parse_args(["--single-stage-alm-X", "V"])` -> `build_single_stage_command(...)`.
  - Acceptance: each parsed `--single-stage-alm-X V` value appears in the emitted subprocess command as `--alm-X V`.

## Priority 2: Missing Regression Tests

- [ ] Add a Phase 3 boundary test: a 1-4% objective improvement at fixed feasibility counts as progress.
  - Acceptance: a regression to the old 5% threshold fails the test.

- [ ] Add a Phase 3 sequence test for `dual_update`, `dual_update`, `dual_update`, `penalty_increase`.
  - Acceptance: stall classification uses current post-iteration progress, not stale pre-iteration state.

- [ ] Add a Phase 3 monotonicity test.
  - Sequence: `dual_update` -> `penalty_increase` -> `dual_update`.
  - Acceptance: both `update_feasibility_tol` and `update_stationarity_tol` are nonincreasing across the sequence.

- [ ] Add a Phase 5 strict-field negative test.
  - Direct call: `_extract_constraint_state({"constraint_values": [...]})`.
  - Acceptance: missing `dual_update_values` raises `KeyError`.

- [ ] Add a long no-progress tolerance acceptance test.
  - Target behavior: the `update_stationarity_tol = min(...)` tighten-only path exits with a meaningful ALM termination reason.
  - Acceptance: the test does not pass merely by exhausting `max_outer_iterations`.

- [ ] Add a physics env preservation test for `run_command`.
  - Inputs: inherited `BOOZER_I`, `PLASMA_CURRENT_A`, and `PROXY_PLASMA_CURRENT_A`.
  - Acceptance: ALM-prefixed variables are stripped by default, but physics env vars survive into the subprocess environment.

- [ ] Add an explicit `inherit_alm_env=True` test for `run_command`.
  - Acceptance: an inherited `ALM_PENALTY_INIT` survives only when `inherit_alm_env=True`.

## Priority 3: User-Facing ALM Contract Docs

- [ ] Add `## ALM Contract` to `examples/single_stage_optimization/HARDWARE_CONSTRAINTS.md`.
  - Keep this in the existing one-stop hardware/ALM operator reference instead of creating another doc.
  - Acceptance: this tracked doc becomes the contract page that any conflicting session memory can point to.

- [ ] Document scalar-only ALM control.
  - Acceptance: no user-facing doc implies per-block penalty control is still configurable.

- [ ] Document `constraint_blocks` as diagnostic labels only.
  - Acceptance: labels are not described as solver-control groups.

- [ ] Document `alm_block_penalties=None` as legacy schema compatibility.
  - Acceptance: readers know `None` is intentional, not missing computation.

- [ ] Document strict evaluator signal field requirements.
  - Required normalized fields: `constraint_values`, `feasibility_values`, `dual_update_values`.
  - Stage-2 signal mode also requires explicit hard/surrogate/hard-dual fields where consumed.
  - All ALM signal arrays must match `constraint_values.shape`; shape mismatch is a contract error.

- [ ] Document checkpoint resume safety.
  - Acceptance: older checkpoints without `constraint_names` are explicitly unsafe for multiplier reuse.

- [ ] Add a grep-style documentation gate.
  - Command: `git grep -nE "block.penalty.*(enable|configur|control)" examples/single_stage_optimization/HARDWARE_CONSTRAINTS.md`.
  - Acceptance: the command returns zero hits.

## Priority 4: Numeric Boundary Documentation

- [ ] Lock `_INFEASIBLE_STALL_OBJECTIVE_RTOL = 1e-6` with a noise-boundary test.
  - Preferred test: an objective drop of `1e-12 * |current_total|` is classified as noise and does not count as progress.
  - Acceptance: noise-scale objective movement remains infeasible stall.
  - Note: the Phase 3 1-4% improvement test differentiates the new contract from the old 5% progress threshold; this test locks the noise-rejection edge.

- [ ] Add a lower-edge progress boundary case if the RTOL behavior remains hard to review.
  - Candidate: an objective drop of `2 * _INFEASIBLE_STALL_OBJECTIVE_RTOL * |current_total|` counts as progress when feasibility is unchanged.
  - Acceptance: accidentally loosening the RTOL by orders of magnitude fails a targeted test.

- [ ] Add a short rationale comment near `_INFEASIBLE_STALL_OBJECTIVE_RTOL`.
  - Acceptance: readers can distinguish noise rejection from the old 5% progress threshold.

## Priority 5: Smoke Gate

- [ ] Decide the smoke gate path.
  - Option A: fetch or restore the missing VMEC equilibrium and re-run the smoke command below.
  - Option B: document the missing equilibrium as an accepted out-of-scope blocker under the original plan's Phase 10 acceptance.
  - Missing path: `/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA/wout_nfp22ginsburg_000_014417_iota15.nc`.
  - Acceptance: final implementation note records either the completed smoke result or the exact missing path, command, and that execution did not reach ALM/single-stage.

Smoke command seed:

```bash
.venv/bin/python examples/single_stage_optimization/run_finite_current_smoke.py \
  --output-root /tmp/simsopt_surrogate_finite_current_smoke_20260507 \
  --stage2-output-root /tmp/simsopt_surrogate_stage2_smoke_20260507 \
  --summary-json /tmp/simsopt_surrogate_finite_current_smoke_20260507/smoke_summary.json \
  --currents-A 0 \
  --nphi 5 \
  --ntheta 4 \
  --mpol 2 \
  --ntor 2 \
  --stage2-timeout-seconds 120 \
  --single-stage-timeout-seconds 120
```

## Commit Grouping

- [ ] Commit 1: off-spec deletion/deprecation scope and memory/doc alignment.
- [ ] Commit 2: ALM CLI SSOT refactor plus schema and round-trip parity tests.
- [ ] Commit 3: missing ALM regression tests, tolerance monotonicity, and runner env tests.
- [ ] Commit 4: ALM contract documentation and smoke-gate note.
- [ ] Commit 5: RTOL boundary coverage and rationale comment.

## Validation Runner

Run the existing ALM and workflow helper suites plus the new targeted tests added by this tracker.

```bash
.venv/bin/python -m pytest -q tests/geo/test_alm_utils.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_workflow_helpers.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_alm_integration.py
.venv/bin/python -m pytest -q tests/geo/test_single_stage_example.py \
  -k "AlmUtilsTests or build_alm_final_constraint_payload or alm_result_view_from_search_eval or stage2_main_alm_path_uses_minimize_alm or validate_resume_alm_state or current_solver_checkpoint_alm_state"
.venv/bin/python -m pytest -q tests/geo/test_constraint_contract.py tests/geo/test_banana_helper_modules.py
git diff --check
```

If new files are introduced, add them to this runner explicitly before closing the tracker.

## Final Acceptance

- [ ] Off-spec policy conflict resolved or explicitly documented.
- [ ] Conflicting session memory is revised or points to the tracked contract doc.
- [ ] One ALM CLI source of truth feeds both parser defaults and command emission.
- [ ] ALM CLI schema parity and parser-to-command round-trip parity are tested.
- [ ] All Phase 3, Phase 5, and Phase 8 plan-required tests exist.
- [ ] Long no-progress tolerance behavior is covered by a targeted test.
- [ ] Runner env tests cover both physics env preservation and `inherit_alm_env=True`.
- [ ] `HARDWARE_CONSTRAINTS.md` contains the discoverable ALM contract.
- [ ] `_INFEASIBLE_STALL_OBJECTIVE_RTOL` has boundary coverage and a rationale comment.
- [ ] Finite-current smoke is run, or its missing-fixture blocker is documented with path and command.
- [ ] Any deferred priority-tracker item is marked explicitly here before merge.
