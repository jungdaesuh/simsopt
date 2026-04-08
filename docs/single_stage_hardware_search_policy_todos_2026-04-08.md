# Single-Stage Hardware Search Policy Todos

Date: 2026-04-08

Scope: replace the ad hoc single-stage search-time hardware gate with an explicit, testable policy while keeping final hardware certification hard and avoiding double-counting existing hardware penalties/ALM residuals.

## Baseline

- [ ] Revert the current local warning-only edit in [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py) so implementation starts from the committed hard-reject baseline.
- [ ] Confirm and record the intended default search policy:
  - [ ] `hard` for backward-compatible default behavior
  - [ ] `adaptive` if continuation-friendly behavior is intended as the new default
- [ ] Decide whether autoresearch-specific behavior belongs in this repo or only in an external wrapper/config layer.

## SSOT Boundaries

- [ ] Keep [evaluate_single_stage_hardware_snapshot()](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_geometry.py) as the hardware-measurement SSOT.
- [ ] Reuse existing continuation state as the search-policy inputs:
  - [ ] `run_dict["accepted_iterations"]`
  - [ ] `search_gate["gate_scale"]`
- [ ] Do not introduce a second hardware objective term on top of existing weighted penalties / ALM residuals.
- [ ] Keep final hardware certification hard in [examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py).

## New Policy Module

- [ ] Add [examples/single_stage_optimization/banana_opt/single_stage_search_policy.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/banana_opt/single_stage_search_policy.py).
- [ ] Add immutable dataclasses only:
  - [ ] `HardwareSearchPolicy`
  - [ ] `SearchContext`
  - [ ] `SearchDecision`
- [ ] Add one pure policy function:
  - [ ] `decide_hardware_search_action(...)`
- [ ] Add one pure helper for rejection magnitude if needed:
  - [ ] `hardware_rejection_increment(previous_objective)`
- [ ] Keep the new module free of SIMSOPT imports and runtime mutation.

## CLI And Config

- [ ] Add `--hardware-search-mode` to the single-stage CLI.
- [ ] Add `--hardware-search-soft-iterations` to the single-stage CLI.
- [ ] Validate both new CLI fields alongside the existing argument validation.
- [ ] Thread both fields through [RunIdentityConfig](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py).
- [ ] Thread both fields through [make_run_identity_config(...)](/Users/suhjungdae/code/columbia/simsopt-surrogate/examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py).
- [ ] Add both fields to the results payload.

## Search-Step Wiring

- [ ] Replace the search-time hardware gate block in `evaluate_search_step()` semantically, not by brittle line number.
- [ ] Keep hard rejection for truly invalid states only:
  - [ ] failed Boozer solve
  - [ ] self-intersection
  - [ ] invalid geometry / nesting / impossible stack state
  - [ ] broken derivatives / NaNs / unusable evaluations
- [ ] Keep topology-gate handling separate from catastrophic-invalid-state handling.
- [ ] Preserve the existing topology modeled-gate behavior:
  - [ ] topology gate failure remains a search-time modeled rejection with its own rejection increment
  - [ ] final topology reporting remains separate from final optimizer success
- [ ] For realized hardware status, route the decision through `decide_hardware_search_action(...)`.
- [ ] Feed the policy helper:
  - [ ] `hardware_status`
  - [ ] `run_dict["accepted_iterations"]`
  - [ ] `search_gate["gate_scale"]`
  - [ ] `run_dict["J"]`
- [ ] If the decision is reject:
  - [ ] set `success = False`
  - [ ] set `rejection_increment`
- [ ] If the decision is warn:
  - [ ] keep the trial step evaluable
  - [ ] print hardware warnings only

## Run-State Cleanup

- [ ] Stop overloading `run_dict["hardware_constraint_status"]`.
- [ ] Introduce:
  - [ ] `run_dict["trial_hardware_status"]`
  - [ ] `run_dict["accepted_hardware_status"]`
- [ ] Initialize the new status keys in the single-stage run-state setup.
- [ ] Update every existing `hardware_constraint_status` consumer:
  - [ ] failure-path logging currently reading the trial status
  - [ ] accepted-iteration callback storage
  - [ ] iteration summary reporting
- [ ] Remove or deprecate the old `hardware_constraint_status` key once all reads/writes are migrated.

## Tests

- [ ] Rewrite the existing hard-mode regression test for rejecting hardware-invalid candidates so it asserts the migrated status keys instead of `run_dict["hardware_constraint_status"]`.
- [ ] Add pure unit tests for `single_stage_search_policy.py`.
- [ ] Add a `warn` mode test:
  - [ ] hardware-invalid trial is not rejected by search policy
  - [ ] hardware status is still recorded
- [ ] Add `adaptive` mode tests:
  - [ ] early continuation / soft window allows a hardware-invalid trial
  - [ ] later phase rejects a hardware-invalid trial
- [ ] Add “catastrophic invalid state always rejects” tests.
- [ ] Add “final endpoint still fails when hardware-invalid” tests.
- [ ] Add CLI parse tests for:
  - [ ] `--hardware-search-mode`
  - [ ] `--hardware-search-soft-iterations`
- [ ] Add results-contract tests for:
  - [ ] `HARDWARE_SEARCH_MODE`
  - [ ] `HARDWARE_SEARCH_SOFT_ITERATIONS`
- [ ] Add run-identity coverage so the new flags affect hashing/identity as intended.

## Validation

- [ ] Run `python -m py_compile` on all touched files.
- [ ] Run `git diff --check` on all touched files.
- [ ] Run the new pure policy-module unit-test file or focused policy slice.
- [ ] Run focused single-stage tests.
- [ ] Run [tests/geo/test_banana_objective_modules.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_banana_objective_modules.py).
- [ ] Run [tests/geo/test_single_stage_example.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_single_stage_example.py).
- [ ] Run [tests/geo/test_single_stage_alm_integration.py](/Users/suhjungdae/code/columbia/simsopt-surrogate/tests/geo/test_single_stage_alm_integration.py).

## Explicit Non-Goals

- [ ] Do not add a second parallel `hardware_violation_loss`.
- [ ] Do not build a generic multi-optimizer constraint-policy framework in this pass.
- [ ] Do not unify Stage 2 and single-stage policy code in this pass unless Stage 2 needs the same abstraction immediately.
- [ ] Do not implement a full restoration phase in this pass.
- [ ] Do not encode policy in comments like "Do not revert" instead of config + tests.
