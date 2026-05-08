# Adversarial Review - ALM Audit

## Model/effort verification (PART A)

`codex --version` output:

```text
WARNING: proceeding, even though we could not update PATH: Operation not permitted (os error 1)
codex-cli 0.129.0
```

`codex exec --help` output relevant to config/model/sandbox:

```text
WARNING: proceeding, even though we could not update PATH: Operation not permitted (os error 1)
Run Codex non-interactively

Usage: codex exec [OPTIONS] [PROMPT]
       codex exec [OPTIONS] <COMMAND> [ARGS]

Commands:
  resume  Resume a previous session by id or pick the most recent with --last
  review  Run a code review against the current repository
  help    Print this message or the help of the given subcommand(s)

Arguments:
  [PROMPT]
          Initial instructions for the agent. If not provided as an argument (or if `-` is used),
          instructions are read from stdin. If stdin is piped and a prompt is also provided, stdin
          is appended as a `<stdin>` block

Options:
  -c, --config <key=value>
          Override a configuration value that would otherwise be loaded from `~/.codex/config.toml`.
          Use a dotted path (`foo.bar.baz`) to override nested values. The `value` portion is parsed
          as TOML. If it fails to parse as TOML, the raw string is used as a literal.
          
          Examples: - `-c model="o3"` - `-c 'sandbox_permissions=["disk-full-read-access"]'` - `-c
          shell_environment_policy.inherit=all`

  -m, --model <MODEL>
          Model the agent should use

  -s, --sandbox <SANDBOX_MODE>
          Select the sandbox policy to use when executing model-generated shell commands
          
          [possible values: read-only, workspace-write, danger-full-access]
```

The help text does not enumerate accepted `reasoning.effort` values. The definitive availability check is `codex debug models`; its output includes this exact JSON excerpt:

```text
{"models":[{"slug":"gpt-5.5","display_name":"GPT-5.5","description":"Frontier model for complex coding, research, and real-world work.","default_reasoning_level":"medium","supported_reasoning_levels":[{"effort":"low","description":"Fast responses with lighter reasoning"},{"effort":"medium","description":"Balances speed and reasoning depth for everyday tasks"},{"effort":"high","description":"Greater reasoning depth for complex problems"},{"effort":"xhigh","description":"Extra high reasoning depth for complex problems"}],
```

Conclusion: `gpt-5.5` is available, quoting the slug `"gpt-5.5"`. `xhigh` is available, quoting the supported effort slug `"xhigh"` from `supported_reasoning_levels`. No fallback was required. If either had been unavailable, the highest available fallback in the same models output would have been the next frontier/coding model with the highest supported reasoning tier, but that fallback was not used.

## Codex invocation

Working tree: `/Users/suhjungdae/code/columbia/simsopt-surrogate`.

Branch check: `git branch --show-current` returned `surrogate-confinement-v2`.

Audit inputs found with `rg --files .alm_audit`:

```text
.alm_audit/algorithm_review.md
.alm_audit/normalization_review.md
.alm_audit/numerics_review.md
.alm_audit/test_coverage_review.md
.alm_audit/regression_review.md
.alm_audit/codex_independent_review.md
.alm_audit/math_review.md
.alm_audit/physics_review.md
```

Review mode: current-tree adversarial code walk in the active Codex session. I did not run a separate `codex exec review` helper process because the requested output depends on writing a single workspace report with exact file:line evidence from the current dirty tree.

Commands run for the requested checks included:

```text
codex --version
codex exec --help
codex debug models
git branch --show-current
git status --short
rg --files .alm_audit
rg -n -i 'kkt|stationarity|feasibility_tol' tests/
```

## Verdict table

| ID | original severity | verdict | one-line reason | new severity |
|---|---:|---|---|---:|
| HIGH-1 | HIGH | PARTIAL | Surrogate inner objective plus hard dual-update signal is real, but `signal_mismatch_active` blocks the converged label. | MEDIUM |
| HIGH-2 | HIGH | CONFIRMED | Direct ALM API path can exhaust final outer with latest action `subproblem_continue` and no `outer_termination`. | MEDIUM |
| HIGH-3 | HIGH | CONFIRMED | Subproblem limit can break outer without lambda/mu change, and staged `gtol` is persisted into later inner options. | MEDIUM |
| HIGH-4 | HIGH | CONFIRMED | Negative ALM thresholds are accepted and then scale-floored to `1e-12` without source-side positivity checks. | HIGH |
| SUB-5 | SUB-MAJOR | CONFIRMED | Default single-stage weighted objective and ALM both penalize the same coil-length threshold. | SUB-MAJOR |
| MED-1 | MEDIUM | PARTIAL | Literal penalty-cap claim is wrong; actual issue is multiplier-cap binding not gating later `converged`. | MEDIUM |
| MED-2 | MEDIUM | CONFIRMED | Pre-inner routing uses the effective feasibility gate while post-inner routing uses raw update tolerance. | MEDIUM |
| MED-3 | MEDIUM | CONFIRMED | `_kkt_stationarity_norm` calls `scipy.optimize.nnls` without local error handling. | MEDIUM |
| MED-4 | MEDIUM | CONFIRMED | No-blocks metadata attachment returns the original evaluation dict while blocks path shallow-copies. | LOW |
| MED-5 | MEDIUM | CONFIRMED | Callback contract says history is borrowed/read-only, but the single-stage callback mutates `history[-1]`. | LOW |
| MED-6 | MEDIUM | CONFIRMED | Scale floors are applied numerically but not encoded in `ALMConstraintMetadata.source`. | MEDIUM |

## Confirmed-finding repros

### HIGH-1: surrogate-vs-hard mismatch in inner-solve ALM is PARTIAL

Confirmed hybrid signal path:

- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1943` calls `augmented_inequality_objective` with `normalized_surrogate_signed_constraint_values`.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1955` stores `"dual_update_values": normalized_dual_update_values`.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:1962` stores `"hard_dual_update_values": normalized_hard_signed_constraint_values`.
- `examples/single_stage_optimization/alm_utils.py:1915` selects `evaluation["hard_dual_update_values"]` as preferred dual-update values for explicit Stage 2 signals.
- `examples/single_stage_optimization/alm_utils.py:2946` passes `routing_state.signal_state.preferred_dual_update_values` into the nonnegative multiplier update.

Refuted unsafe-convergence part:

- `examples/single_stage_optimization/alm_utils.py:2005` sets `signal_mismatch_active` when hard and surrogate active masks disagree.
- `examples/single_stage_optimization/alm_utils.py:2020` also activates mismatch on direct boundary disagreement.
- `examples/single_stage_optimization/alm_utils.py:4091` through `examples/single_stage_optimization/alm_utils.py:4096` require `not signal_mismatch_active` for the `converged` action.

Reachable behavior: a Stage 2 evaluation can optimize surrogate signed constraints while updating multipliers on hard signed constraints. The current tree avoids labeling such a mismatch converged, so the remaining confirmed risk is stall/failure or misleading progress semantics, not false hard-constraint certification.

### HIGH-2: final-outer `subproblem_continue` termination is CONFIRMED

Evidence:

- `examples/single_stage_optimization/alm_utils.py:3457` defines `_emit_alm_subproblem_continue` without any `is_final_outer` argument or `outer_termination` annotation.
- `examples/single_stage_optimization/alm_utils.py:4219` through `examples/single_stage_optimization/alm_utils.py:4226` call `_emit_alm_subproblem_continue` from the `signal_mismatch_active` progress path.
- `examples/single_stage_optimization/alm_utils.py:4405` iterates `range(settings.max_subproblem_continuations + 1)`.
- `examples/single_stage_optimization/alm_utils.py:4454` through `examples/single_stage_optimization/alm_utils.py:4456` return `EXHAUST` on loop exhaustion for the final outer iteration.
- `examples/single_stage_optimization/alm_utils.py:1692` through `examples/single_stage_optimization/alm_utils.py:1695` return the latest action string if no `outer_termination` is present.
- `examples/single_stage_optimization/alm_utils.py:4577` through `examples/single_stage_optimization/alm_utils.py:4581` use that helper to label the final failure result.

Reachable repro scenario: call the direct ALM API with `ALMSettings(max_outer_iterations=1, max_subproblem_continuations=0)` and a Stage 2 style evaluation whose inner solve makes progress while `signal_mismatch_active` remains true. The single continuation emits `action="subproblem_continue"`, final outer exhausts, and failure termination is labeled `subproblem_continue` rather than `max_outer_*`.

CLI split: `examples/single_stage_optimization/alm_utils.py:321` through `examples/single_stage_optimization/alm_utils.py:323` reject `max_subproblem_continuations <= 0` in the CLI validation path, so the exact `0` setting is not CLI-reachable through that validator. It remains API-reachable because `ALMSettings` itself accepts the value.

### HIGH-3: subproblem-limit and staged-gtol drift are CONFIRMED

Subproblem-limit evidence:

- `examples/single_stage_optimization/alm_utils.py:4267` begins the hard-feasible subproblem-limit branch.
- `examples/single_stage_optimization/alm_utils.py:4282` through `examples/single_stage_optimization/alm_utils.py:4298` set action `subproblem_limit` and emit a snapshot.
- `examples/single_stage_optimization/alm_utils.py:4312` through `examples/single_stage_optimization/alm_utils.py:4317` return `BREAK_OUTER` without calling the dual update or penalty increase transition.
- `examples/single_stage_optimization/alm_utils.py:4433` through `examples/single_stage_optimization/alm_utils.py:4436` carry the returned state out of the inner iteration.
- `examples/single_stage_optimization/alm_utils.py:4553` through `examples/single_stage_optimization/alm_utils.py:4556` copy that state into the outer loop.

Staged-gtol evidence:

- `examples/single_stage_optimization/alm_utils.py:1628` copies `inner_options`.
- `examples/single_stage_optimization/alm_utils.py:1638` computes `staged_gtol`.
- `examples/single_stage_optimization/alm_utils.py:1640` writes `options["gtol"] = max(base_gtol, staged_gtol)`.
- `examples/single_stage_optimization/alm_utils.py:4001` persists `inner_attempt.last_inner_options` back into `state.inner_options`.

Reachable repro scenario: an ALM run that is hard-feasible for update but not stationarity-converged and reaches `max_subproblem_continuations` exits the outer step with the same multipliers and penalty. The next outer step also inherits a staged `gtol` as the new base, so later tightening is monotonic-limited by a previous staged value.

### HIGH-4: negative ALM thresholds floored to `1e-12` are CONFIRMED

CLI and validation evidence:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1383` through `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1413` define the ALM threshold CLI args as plain `type=float`.
- `examples/single_stage_optimization/alm_utils.py:318` through `examples/single_stage_optimization/alm_utils.py:359` validate ALM iteration and penalty/tolerance knobs but do not validate ALM threshold positivity.

Single-stage metadata evidence:

- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:495` through `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:503` convert the threshold to float and compute `scale=max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR)`.
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:507` records the source as `threshold:<name>`, not as a floor-adjusted source.

Stage 2 metadata evidence:

- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:694` through `examples/single_stage_optimization/banana_opt/stage2_objectives.py:699` require only presence, then return `float(value)`.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:724` computes `scale=max(raw_threshold, ALM_OBJECTIVE_SCALE_FLOOR)`.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:728` records source `"stage2_iota_penalty_threshold"`.

Reachable repro scenario: `--alm-formulation thresholded_physics --alm-qs-threshold -1.0` passes argument parsing and the existing ALM CLI validator, then produces metadata with raw threshold `-1.0` and scale `1e-12`. Any residual normalized by that scale is inflated by up to `1e12` relative to unit-scale semantics.

### SUB-5: weighted_sum plus ALM double-feeds coil-length penalty is CONFIRMED

Evidence:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1472` through `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1473` set `--length-weight` default to `1`.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3706` builds `JCurveLength = QuadraticPenalty(curvelength, length_target, "max")`.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3720` through `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3729` pass `LENGTH_WEIGHT` and `JCurveLength` into the weighted total objective.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4261` through `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4268` also pass the same `curvelength` and `length_target` into the ALM objective path.
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:781` through `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:786` add the ALM `coil_length_upper_bound` constraint from those values.

Reachable repro scenario: default single-stage ALM with `length_weight` unset uses the weighted base objective's `LENGTH_WEIGHT * QuadraticPenalty` and also enforces the same length target through the ALM inequality. This is not merely duplicated metadata; both paths contribute to the solve objective.

### MED-1: multiplier-cap binding is PARTIAL, literal penalty-cap wording is REFUTED

Confirmed multiplier-cap evidence:

- `examples/single_stage_optimization/alm_utils.py:1556` through `examples/single_stage_optimization/alm_utils.py:1575` clamp multiplier updates and report cap-binding flags.
- `examples/single_stage_optimization/alm_utils.py:4248` through `examples/single_stage_optimization/alm_utils.py:4251` propagate `cap_binding_detected`.
- `examples/single_stage_optimization/alm_utils.py:4091` through `examples/single_stage_optimization/alm_utils.py:4096` do not include cap-binding status in the `converged` gate.
- `examples/single_stage_optimization/alm_utils.py:2765` through `examples/single_stage_optimization/alm_utils.py:2766` record multiplier-cap diagnostics in the final result.

Refuted literal penalty-cap part:

- `examples/single_stage_optimization/alm_utils.py:2970` through `examples/single_stage_optimization/alm_utils.py:3008` build a failure result for true penalty-cap termination.
- `examples/single_stage_optimization/alm_utils.py:2997` sets `"termination_reason": "penalty_cap_reached"`.

Result: the audit item cited multiplier-cap code while calling it penalty-cap behavior. The confirmed issue is that multiplier-cap binding is diagnostic-only and can coexist with a later `converged` label.

### MED-2: routing tolerance divergence is CONFIRMED

Evidence:

- `examples/single_stage_optimization/alm_utils.py:3776` through `examples/single_stage_optimization/alm_utils.py:3778` compute `effective_feasibility_tol`.
- `examples/single_stage_optimization/alm_utils.py:3785` through `examples/single_stage_optimization/alm_utils.py:3790` pass that effective gate into `current_routing_state`.
- `examples/single_stage_optimization/alm_utils.py:3936` through `examples/single_stage_optimization/alm_utils.py:3941` pass raw `state.update_feasibility_tol` into post-inner `routing_state`.
- `examples/single_stage_optimization/alm_utils.py:4080` through `examples/single_stage_optimization/alm_utils.py:4082` later uses the effective gate again for `hard_feasible_for_update`.

Reachable repro scenario: when the feasibility gate is clamped or otherwise differs from `state.update_feasibility_tol`, pre-inner routing, post-inner routing, and hard-feasible update decisions can classify the same values under different tolerances in the same outer iteration.

### MED-3: `_kkt_stationarity_norm` unguarded `nnls` call is CONFIRMED

Evidence:

- `examples/single_stage_optimization/alm_utils.py:2041` defines `_kkt_stationarity_norm`.
- `examples/single_stage_optimization/alm_utils.py:2056` through `examples/single_stage_optimization/alm_utils.py:2065` build active constraint gradients.
- `examples/single_stage_optimization/alm_utils.py:2071` calls `nnls(active_matrix, -total_grad_array)` with no local `try`/`except` or diagnostic fallback.
- `examples/single_stage_optimization/alm_utils.py:2103` through `examples/single_stage_optimization/alm_utils.py:2111` call this path from `_stationarity_metrics` whenever signal mismatch is not active.

Reachable repro scenario: a malformed or numerically pathological active constraint matrix can raise from SciPy `nnls` during stationarity diagnostics and abort the ALM path instead of returning a diagnostic failure result. This is not a recommendation to add broad defensive code; it is a narrow crash surface in a diagnostic helper used on the main path.

### MED-4: no-blocks metadata aliasing is CONFIRMED

Evidence:

- `examples/single_stage_optimization/alm_utils.py:2204` defines `_attach_alm_constraint_metadata`.
- `examples/single_stage_optimization/alm_utils.py:2209` through `examples/single_stage_optimization/alm_utils.py:2210` return the original `evaluation` dict when `constraint_blocks_tuple is None`.
- `examples/single_stage_optimization/alm_utils.py:2211` through `examples/single_stage_optimization/alm_utils.py:2214` shallow-copy the dict when blocks are present.

Result: the aliasing contract differs by metadata mode. The direct no-blocks lane can expose later callers to original evaluation dict mutations, while the blocks lane at least isolates top-level keys.

### MED-5: live history callback mutation is CONFIRMED

Evidence:

- `examples/single_stage_optimization/alm_utils.py:2379` through `examples/single_stage_optimization/alm_utils.py:2385` document that history is borrowed/read-only ALM state and pass `history` to the callback.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:9015` stores the live `history` object in `alm_partial_state`.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:9020` through `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:9021` mutate `history[-1]["smoothing_changed"]`.
- `tests/geo/test_alm_utils.py:2390` explicitly asserts the callback receives the live history object by identity.
- `tests/geo/test_alm_utils.py:2400` through `tests/geo/test_alm_utils.py:2403` only protect `latest_entry` mutation, not mutation through `history[-1]`.

Result: the widened callback contract is real. The current single-stage caller uses that widened contract to mutate ALM history after emission.

### MED-6: scale floor is not reflected in metadata source is CONFIRMED

Evidence:

- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:503` applies `ALM_OBJECTIVE_SCALE_FLOOR`.
- `examples/single_stage_optimization/banana_opt/single_stage_objectives.py:507` records source `threshold:<name>` without indicating the floor.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:724` applies `ALM_OBJECTIVE_SCALE_FLOOR`.
- `examples/single_stage_optimization/banana_opt/stage2_objectives.py:728` records source `"stage2_iota_penalty_threshold"` without indicating the floor.

Result: history and metadata consumers see the nominal threshold source, not the effective floor-based scale source. That makes post-run scale provenance ambiguous exactly in the pathological negative/near-zero threshold cases.

## Refuted-finding evidence

No listed item is fully refuted.

Refuted portions:

- HIGH-1 false convergence is refuted by `examples/single_stage_optimization/alm_utils.py:4091` through `examples/single_stage_optimization/alm_utils.py:4096`, which include `not signal_mismatch_active` in the converged gate.
- HIGH-2 exact `max_subproblem_continuations==0` CLI path is refuted by `examples/single_stage_optimization/alm_utils.py:321` through `examples/single_stage_optimization/alm_utils.py:323`, but the direct API path remains confirmed.
- MED-1 literal penalty-cap wording is refuted by `examples/single_stage_optimization/alm_utils.py:2970` through `examples/single_stage_optimization/alm_utils.py:3008`, where true penalty-cap termination returns a failure result labeled `penalty_cap_reached`.

## Partial-finding splits

HIGH-1 split:

- CONFIRMED: inner augmented objective can be surrogate-based while ALM multiplier updates use hard dual-update values.
- CONFIRMED: signal mismatch is explicitly detected from hard/surrogate mask disagreement and direct boundary disagreement.
- REFUTED: current convergence guard does not accept `converged` while `signal_mismatch_active` is true.
- Residual severity: MEDIUM because the live risk is continuation/stall/failure labeling under mismatched signals, not an immediate false success label.

MED-1 split:

- REFUTED: true penalty-cap termination is not mislabeled as `converged`.
- CONFIRMED: multiplier-cap binding is not part of the converged gate and is only recorded diagnostically.
- Residual severity: MEDIUM because a result can satisfy local feasibility/stationarity gates while carrying a capped multiplier state that weakens KKT interpretation.

## Missed-by-agents (1-3 findings)

### NEW-MED-1: resume ALM accepts nonfinite or negative multipliers

Severity: MEDIUM.

Evidence:

- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4342` defines `validate_resume_alm_state`.
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4360` through `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4366` check multiplier shape but not finite values and not nonnegativity.
- `examples/single_stage_optimization/alm_utils.py:2171` through `examples/single_stage_optimization/alm_utils.py:2175` copy initial multipliers without finite or nonnegative validation.
- `tests/geo/test_single_stage_example.py:11494` through `tests/geo/test_single_stage_example.py:11536` cover exact names, length mismatch, and missing names, but not nonfinite or negative multipliers.

Reachable repro scenario: a resume JSON with correct constraint names and correct multiplier length but a negative or `NaN` multiplier passes the resume validator into ALM initialization. Penalty positivity is later checked, but multiplier validity is not.

### NEW-LOW-2: accepted no-blocks evaluation can alias mutable caller buffers through the cache

Severity: LOW.

Evidence:

- `examples/single_stage_optimization/alm_utils.py:184` stores `self.cached_evaluation = evaluation` in `_ALMInnerAttemptEvaluator.fun`.
- `examples/single_stage_optimization/alm_utils.py:3105` through `examples/single_stage_optimization/alm_utils.py:3111` reuse `evaluator.cached_evaluation` for the candidate when it matches the optimizer result.
- `examples/single_stage_optimization/alm_utils.py:3157` through `examples/single_stage_optimization/alm_utils.py:3161` attach metadata to that accepted candidate evaluation.
- `examples/single_stage_optimization/alm_utils.py:2209` through `examples/single_stage_optimization/alm_utils.py:2210` return the original dict in the no-blocks lane.
- `examples/single_stage_optimization/alm_utils.py:495` through `examples/single_stage_optimization/alm_utils.py:510` use `np.asarray` when building evaluation arrays; `np.asarray` does not copy an already-compatible mutable array.

Reachable repro scenario: a custom `evaluate_problem` implementation using reusable NumPy buffers and no constraint blocks can leave the accepted evaluation dict and arrays aliased to caller-owned mutable buffers. The production helper often creates fresh arrays, so this is lower severity, but the public ALM API does not enforce the stronger ownership contract.

## Test coverage cross-check

Command run:

```text
rg -n -i 'kkt|stationarity|feasibility_tol' tests/
```

Verdict on the `test_coverage_review.md` claim: CONFIRMED with nuance. There are KKT and stationarity tests, but I did not find a nontrivial test that verifies KKT on a successful final iterate with active constraints.

Existing coverage:

- `tests/geo/test_alm_utils.py:1712` defines `test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint`.
- `tests/geo/test_alm_utils.py:1744` asserts `self.assertFalse(result.success)`.
- `tests/geo/test_alm_utils.py:1747` through `tests/geo/test_alm_utils.py:1748` assert `final_raw_stationarity_norm == 1.0` and `final_kkt_stationarity_norm == 0.0`.
- `tests/geo/test_single_stage_example.py:11576` through `tests/geo/test_single_stage_example.py:11612` duplicate the same failure-result KKT diagnostic pattern.
- `tests/geo/test_alm_utils.py:3451` through `tests/geo/test_alm_utils.py:3494` verify KKT stationarity at a nearly active boundary, but the result is still `success=False` with termination `max_outer_after_dual_update`.
- `tests/geo/test_alm_utils.py:3669` through `tests/geo/test_alm_utils.py:3712` verify callback interruption when the KKT gate is hit, again as history/control-flow behavior rather than successful final KKT certification.
- `tests/geo/test_alm_utils.py:4397` through `tests/geo/test_alm_utils.py:4442` test `converged` on a trivial strict-interior case where `grad=0`, `constraint_values=-1.0`, and the inner solver is patched to fail if called.

Gap that remains:

- No test exercises a successful final ALM result with an active inequality, nonzero raw objective gradient, nonzero multiplier, and asserted final KKT residual.
- No test covers the Stage 2 hard/surrogate mismatch path all the way to final termination labeling.
- No test covers negative ALM threshold rejection because no such rejection exists.
- No test covers resume-state negative or nonfinite multipliers.

## Final verdict (3 sentences)

The 8-agent audit was not overbroad: most concrete code-path findings survive current-tree tracing, but one HIGH is mitigated by the current signal-mismatch convergence guard and the penalty-cap item was actually a multiplier-cap item. The live tree still has production-relevant issues around threshold validation, final-outer termination labeling, tolerance and `gtol` schedule drift, resume multiplier validation, and length double-feeding. This review is a fail for ALM cleanup readiness, but it is not evidence that the current tree labels hard/surrogate mismatch as converged today.
