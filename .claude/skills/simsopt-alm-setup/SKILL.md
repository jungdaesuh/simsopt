---
name: simsopt-alm-setup
description: "Install simsopt's augmented-Lagrangian solver (simsopt.solve.alm) into a user's simsopt checkout and set up a constrained optimization min f(x) s.t. g_i(x) <= 0 for their case: Stage-2 coils, Boozer single-stage, or a custom problem. Detects the Python and simsopt install and picks an install route (merge the alm-library branch, copy the package, or upstream), interviews the user (objective, constraints, stateful evaluator, hybrid signals), generates a problem module and a runner from tested templates, verifies gradients, constraint signs and scales plus a smoke run, and explains termination reasons. Use when the user says 'set up ALM', 'install the ALM solver', 'constrained coil optimization in simsopt', 'replace penalty weights with constraints', 'minimize_alm', or asks what an ALM termination_reason means or how to tune ALMSettings."
---

# Set up simsopt's ALM solver

This skill installs the augmented-Lagrangian solver `simsopt.solve.alm` into a
simsopt installation and sets up one constrained optimization,
`min f(x) s.t. g_i(x) <= 0`, as two new files built from tested templates,
checked before the real run.

`$SKILL_DIR` is the skill's directory (`.claude/skills/simsopt-alm-setup` in
the simsopt repository). `<python>` is the interpreter the user's
optimization runs with (ask when unclear). `<dir>` is the directory the
generated files go into.

## Rules

- New code goes into new files in `<dir>`, which must not exist yet or be
  empty. An existing file (for example the user's current optimization
  script) changes only through a unified diff shown to the user and applied
  after the user approves it.
- simsopt's own sources change only through the install route of step 1,
  after the user approves its commands. Nothing is pushed.
- Stop and report instead of working around when: `check_env.py` lists
  `blockers`; `smoke_toy.py` fails; a check in step 4 fails for a reason
  outside the generated files; the user declines a step.
- Change a check's threshold or a template's safeguard only when the user
  asks; fix the cause instead.

## 1. Install

1. Run `<python> $SKILL_DIR/scripts/check_env.py --python <python>`, adding
   `--checkout <path>` when the user named their simsopt checkout and
   `--fork-url <url>` when the repository publishing the `alm-library` branch
   is known. Read `route`, `blockers` and `notes` from the `CHECK_ENV` line.
2. If `blockers` is not empty, report them and stop.
3. If `route` is not `ready`, show the user the commands of that route from
   [install.md](references/install.md) with the placeholders filled in, run
   them after the user approves, and rerun `check_env.py`. Repeat until
   `route` is `ready`.
4. Run `<python> $SKILL_DIR/scripts/smoke_toy.py`. Step 1 is done when it
   exits 0 (`"passed": true`).

## 2. Interview

Ask these multiple-choice questions in one message and wait for the answers.
<!-- skill-only -->
Use the AskUserQuestion tool when it is available. Do not generate anything
before the answers arrive.
<!-- /skill-only -->

1. Problem type: (a) Stage 2: coils for a fixed target surface; (b) Boozer
   single-stage: coils optimized on a Boozer surface re-solved each
   evaluation; (c) custom: any f(x) whose value and gradient you can compute.
2. Objective f: (a) the template's (Stage 2: squared flux plus a length
   weight; Boozer: non-quasi-symmetric ratio); (b) another simsopt objective
   (which); (c) custom code (how value and gradient are computed).
3. Constraints: for each, the quantity, the bound type (upper, lower, or
   band), and the threshold with units. Offer the template's rows (Stage 2:
   coil-coil distance, coil-surface distance, maximum curvature, mean squared
   curvature, coil length; Boozer: iota band, major-radius band, coil length,
   coil-coil distance, maximum curvature, mean squared curvature).
   For a coil-length bound, also ask what it bounds: (a) each base coil
   (`PER_BASE_COIL`, one row per base coil); (b) the sum over the base coils
   (`SUM_OF_BASE_COILS`, one row); (c) the sum over all physical coils after
   the symmetry (`SUM_OF_ALL_COILS`, one row). With stellarator symmetry each
   base coil has 2 x nfp physical copies (16 coils from the Stage-2
   template's 4, 18 from NCSX's 3), so the same coils have a (c) length
   2 x nfp times their (b) length: state that multiplicity with the question.
4. Does an evaluation re-solve something warm-started from an earlier
   evaluation (Boozer Newton, a VMEC restart, any inner solve)? (a) no:
   stateless; (b) yes: stateful.
5. Distance and curvature rows: (a) smooth rows only (default, conservative);
   (b) hybrid quartet: the exact values decide feasibility and drive the
   multiplier update.
6. Opt-ins: (a) none; (b) per-step history JSON; (c) checkpoints to resume
   from; (d) both.
7. Where to put `<dir>` (default: a new `alm_<name>/` next to the user's
   optimization script), and whether an existing script should call the new
   files (then step 3.5 applies).

For a custom problem also ask for points where the user knows, from the
physics, that a row is violated or satisfied: they become the sign probes.

## 3. Generate

| Answer | Template and change |
|---|---|
| 1(a) | [stage2.py](templates/stage2.py) |
| 1(b) | [boozer_single_stage.py](templates/boozer_single_stage.py) |
| 1(c) | [generic.py](templates/generic.py) |
| 4(a) | Keep `cached_alm_evaluator(self.physics)`. |
| 4(b) with 1(c) | Evaluate without a cache and warm-start only from accepted solutions: copy the state pattern of the Boozer template (`solve`, `accept_inner_iterate`, `accept_outer_iterate`, `snapshot_accepted`, `restore_incumbent`, and `solver_callbacks` returning all four callbacks). |
| 3, coil length | Set `MAX_LENGTH` (Stage 2) or `LENGTH_MAX` (Boozer) and `LENGTH_SCOPE` to the answer. |
| 5(b) with 1(a) | Set `HYBRID_QUARTET = True`. |
| 5(b) with 1(b) or 1(c) | Add the quartet to `physics` as in [api.md](references/api.md) (Hybrid quartet), exact values from each kernel's third item. |
| 6 | Nothing to generate: `run_alm.py --history FILE` and `--checkpoints DIR`. |

1. Create `<dir>`. Copy the chosen template to `<dir>/alm_problem.py` and
   [run_alm.py](templates/run_alm.py) to `<dir>/run_alm.py`.
2. In `alm_problem.py`, edit the parts marked `SETUP` to the answers: the
   objective, the rows with their thresholds and units, and the sign probes.
   Keep every row in the `g <= 0` convention, divided by its bound or typical
   size so that it is O(1); keep row names unique.
3. Keep the template's `ALMSettings` preset. Change a field only with a
   one-line comment giving the reason ([settings.md](references/settings.md)).
4. Keep the problem-module contract of [api.md](references/api.md) (the
   checks and the runner read it).
5. If an existing script should use the new files, change it by the
   pattern of [existing-script.md](references/existing-script.md) (import
   `build_problem` and `run`, move its setup values into the `SETUP`
   constants, delete its penalty terms and optimizer call), show the unified
   diff of that script, and apply it only after the user approves.

## 4. Verify

Run each from any directory; all three must pass before the real run.

1. `PYTHONPATH=<dir> <python> $SKILL_DIR/scripts/gradient_check.py --smoke`
   must exit 0. It compares each claimed directional derivative with central
   differences over a sweep of relative steps (1 down to 1e-10), and each
   quantity ends in one of:
   - `FAIL`: the step ranges that converge (three steps agreeing to 1e-4)
     agree on one value, and the claim misses it by more than 5e-4 (for a
     zero value, the claim is clearly outside its round-off): the gradient
     of the named quantity in `alm_problem.py` is wrong; fix it.
   - `PASS`: the ranges agree on one value, and two consecutive steps inside
     one of them agree with the claim to 1e-6 (for a zero value, the claim
     is zero to its round-off). Nothing to do.
   - `NOT TESTED`: nothing converges (noise, a kink, or a step range that
     misses the derivative), or ranges converge to different values (a
     float32 or quantized term, a warm-started inner solve with a loose
     tolerance, or a kink: check the evaluator), or the claim is within 5e-4
     but not resolved to 1e-6. Check the quantity at a nearby point, or
     inspect it; there is no step to retry with.
   - `NONFINITE`: the value or the claimed gradient at x0 is not finite.
2. `PYTHONPATH=<dir> <python> $SKILL_DIR/scripts/sign_check.py --smoke` must
   exit 0. On a failure, fix the named row's sign or its probe expectation.
   Fix scale warnings by rescaling rows or f. For coverage warnings, add a
   probe where one is cheap; otherwise tell the user which sides stay
   unchecked.
3. `cd <dir> && <python> run_alm.py --smoke` must print an `ALM_RESULT` line.
   Report its `termination_reason`, `max_violation` and `multipliers`. Any
   termination reason is acceptable for the smoke run; an exception is not.

Then give the user the full-run command,
`cd <dir> && <python> run_alm.py [--history history.json] [--checkpoints checkpoints]`.

## 5. Interpret and tune

1. Look up the run's `termination_reason` in
   [termination.md](references/termination.md) and apply its action. Check
   `restored_best_feasible` too: a restored `x` is feasible and usable.
2. Change one thing per rerun, and name the reason (the table row or the
   pitfall from [pitfalls.md](references/pitfalls.md)).
3. Stop when the run ends `converged` or `constraints_inactive_converged`,
   or when the user accepts a feasible result.

## Files

- `references/`: [install.md](references/install.md) (route commands),
  [api.md](references/api.md) (names, evaluator and problem-module
  contracts), [existing-script.md](references/existing-script.md) (adapting
  a penalty script), [settings.md](references/settings.md) (every `ALMSettings`
  field, inner options), [termination.md](references/termination.md) (every
  termination reason), [pitfalls.md](references/pitfalls.md).
- `templates/`: `generic.py`, `stage2.py`, `boozer_single_stage.py` (problem
  modules) and `run_alm.py` (the runner).
- `scripts/`: `check_env.py`, `smoke_toy.py`, `gradient_check.py`,
  `sign_check.py`, and `build_guide.py`, which generates the human guide
  `docs/alm_setup_guide.md` from these files. After editing any file here,
  run `python $SKILL_DIR/scripts/build_guide.py`.
