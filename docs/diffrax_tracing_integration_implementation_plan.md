# Diffrax Tracing Integration Implementation Plan

## Purpose

This plan defines a staged, reviewable path for evaluating and integrating
Diffrax into the repo's JAX ODE tracing code without destabilizing the current
hand-written integrators, parity tests, or public tracing wrappers.

## Goals

- Add enough structure to evaluate Diffrax against the current JAX tracing
  contracts before making it a production dependency.
- Prove whether Diffrax can reproduce the current fieldline, particle,
  Poincare, and on-axis-iota behavior within explicit parity tolerances.
- Keep the existing JAX tracing API, fixed-shape outputs, status codes, event
  payloads, and adapter behavior stable unless a later decision explicitly
  changes them.
- Make any new Diffrax dependency opt-in until dependency review, parity,
  autodiff, strict-transfer, compile, and memory gates pass.

## Non-Goals

- Do not replace the existing hand-written tracing integrators in the first
  implementation phase.
- Do not use Diffrax for Boozer optimizers, L-BFGS/BFGS, LM, GMRES, QFM,
  permanent-magnet optimization, or algebraic root solves.
- Do not use Diffrax for algebraic curve, framed-curve, finite-build, banana,
  or pairwise geometry kernels.
- Do not change the current fixed-step RK4 semantics or artifact metadata of
  the default Poincare runner until an adaptive Diffrax path has separate
  comparison artifacts.
- Do not add Diffrax as a direct dependency without explicit user approval and
  a completed dependency review.

## Current Context

- `src/simsopt_jax/core/tracing.py` currently implements an in-repo JAX
  Dormand-Prince RK4(5) integrator, PI step controller, fixed-length
  `jax.lax.scan` adaptive driver, FSAL reuse, and bracketed Illinois event
  localization for fieldlines, Cartesian guiding-center, Boozer
  guiding-center, and full-orbit tracing.
- `FieldlineTracingSpec` owns the public compiled tracing contract: `tmax`,
  `rtol`, `atol`, `dtmax`, static `max_steps`, static `max_root_iters`, and
  static `max_phi_hits`. The current result shape includes padded trajectory
  and event buffers with masks and counters.
- `tests/jax/core/test_tracing_jax_item14.py` asserts that the fieldline
  driver lowers to `scan`, not top-level `while`, and supports reverse-mode AD
  through `jax.grad`.
- `src/simsopt_jax_adapters/field/tracing.py` is the public routing boundary
  for `compute_fieldlines`, guiding-center particles, and full-orbit particles.
  It sets hard-coded `max_steps`, `max_phi_hits`, per-lane `dtmaxs`, and
  translates CPU stopping criteria into JAX stopping-criterion payloads.
- `examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces_jax_default.py`
  currently uses a hand-written fixed-step RK4 scan over toroidal angle. Its
  output metadata advertises the fixed-step integrator, so Diffrax belongs
  there first as an adaptive comparison mode, not as a silent replacement.
- `src/simsopt_jax/core/magnetic_axis_helpers.py` contains a separate
  hand-written Dormand-Prince adaptive loop for Greene on-axis iota. Its
  contract supports forward-mode AD and explicitly excludes reverse-mode AD
  through the current `jax.lax.while_loop`.
- `pyproject.toml` currently lists JAX, Optax, Optimistix, Lineax, and Equinox
  in the `JAX` optional dependency group. No `diffrax` dependency or
  production/test code reference exists outside this plan in the current grep
  surface.
- Context7 documentation for `/patrick-kidger/diffrax` shows the relevant API
  shape: `diffeqsolve(ODETerm(...), Dopri5(), PIDController(...), SaveAt(...),
  event=..., adjoint=...)`, with JAX-compatible ODE solves and explicit
  event/adjoint choices.

## Rationale

The current tracing stack is the only part of this repo where Diffrax addresses
the same problem the code is already solving: adaptive ODE integration with
JAX autodiff and batched execution. The value is not just line-count reduction;
it is replacing a custom solver/controller/event stack with a maintained solver
library if, and only if, it can preserve the repo's fixed-shape compiled API and
parity contracts.

The right tradeoff is an opt-in backend behind an internal interface. That keeps
the current solver as the reference implementation while giving a contained way
to test Diffrax on the smallest useful surfaces. Forcing Diffrax into optimizer
or algebraic geometry code would increase abstraction mismatch and compile
surface without removing the actual linear algebra, root finding, or geometry
work.

## Assumptions

- The first useful Diffrax experiment can be scoped to ODE solve behavior
  without changing public tracing wrapper signatures.
- Diffrax can be tested in the same pinned JAX validation environment used by
  this checkout, or the plan stops at dependency feasibility.
- Exact step sequences are not required for acceptance unless a test explicitly
  asserts them; observable endpoints, event payloads, masks, statuses, and
  gradients are the compatibility contract.
- The existing hand-written solver remains the reference backend until a later
  review explicitly promotes Diffrax.
- Any production dependency change requires user approval after dependency
  review, even if the prototype code passes local validation.

## Implementation Plan

1. Establish dependency and design gates.
   - [ ] Run a dependency review for Diffrax: necessity, maintenance health,
     license compatibility, security posture, lockfile or immutable resolution,
     public API exposure, and removal plan.
   - [ ] Decide whether Diffrax lives in the existing `JAX` optional dependency
     group or a narrower opt-in extra such as `JAX_DIFFRAX`.
   - [ ] Record the selected Diffrax version constraint and compatible JAX
     range before writing production code.
   - [ ] Get explicit user approval before adding Diffrax to `pyproject.toml`.
   - [ ] Write a short internal interface comment for the solver boundary before
     implementation. The interface must describe inputs, output shape,
     event/status responsibilities, and adjoint policy in five lines or fewer.

2. Define an internal solver boundary in `src/simsopt_jax/core/tracing.py`.
   - [ ] Separate solver mechanics from tracing payload assembly without
     changing `FieldlineTracingSpec`, `trace_fieldline`, or public result
     dataclasses.
   - [ ] Keep the existing hand-written Dormand-Prince path as the default
     backend and reference oracle.
   - [ ] Add a private typed backend selector that is not exposed through public
     wrappers until parity is proven.
   - [ ] Ensure backend selection is static at JIT boundaries so it does not
     become a runtime data-dependent branch inside compiled kernels.
   - [ ] Preserve current padded trajectory shape, live mask, status code,
     `phi_hits`, `phi_hits_count`, and overflow semantics.

3. Prototype the smallest Diffrax solve surface.
   - [ ] Start with a no-event `trace_fieldline` comparison against a simple
     analytic field, using `diffrax.ODETerm`, `diffrax.Dopri5`,
     `diffrax.PIDController`, and fixed `max_steps`.
   - [ ] Use `diffrax.SaveAt(ts=...)` or equivalent saved output policy only if
     it can be mapped back into the current padded trajectory contract.
   - [ ] Pick and document the adjoint policy explicitly. Reverse-mode AD
     through `trace_fieldline` is part of the current test contract.
   - [ ] Compare endpoint values, accepted status, trajectory mask, and
     reverse-mode gradient against the existing solver on analytic fields.
   - [ ] Reject the prototype if it requires changing public result shapes,
     event buffer schema, or public wrapper signatures.

4. Extend the prototype to fieldline events.
   - [ ] Reproduce phi-plane event capture for `trace_fieldline` before adding
     levelset or stopping-criterion events.
   - [ ] Compare event ordering, event row payloads, `phi_hits_count`, and
     overflow handling against the current Illinois-localized implementation.
   - [ ] Decide whether Diffrax event handling is sufficient, or whether the
     repo should keep current event localization around Diffrax dense output.
   - [ ] Add explicit tests for first-event and earliest-stopping behavior so
     event semantics are not validated only by aggregate endpoint proximity.

5. Extend through the public field tracing adapter.
   - [ ] Add an internal adapter-level route in
     `src/simsopt_jax_adapters/field/tracing.py` only after core fieldline
     parity passes.
   - [ ] Preserve existing hard-coded public limits first: `max_steps=4000`,
     `max_phi_hits=4096`, and per-lane `dtmaxs`.
   - [ ] Confirm translated stopping criteria retain the same supported and
     unsupported class behavior.
   - [ ] Keep existing CPU-style return payload conversion unchanged unless a
     separate API-evolution review approves a public behavior change.

6. Evaluate guiding-center, Boozer guiding-center, and full-orbit tracing.
   - [ ] Port the Cartesian guiding-center RHS comparison after fieldline event
     parity passes.
   - [ ] Port Boozer guiding-center modes only after confirming Diffrax can
     accept the current frozen Boozer field states without host callbacks.
   - [ ] Port full-orbit Lorentz tracing after guiding-center parity passes,
     preserving `max_steps=20000` in the public particle wrapper.
   - [ ] Validate invariants already covered by the full-orbit tests before
     treating endpoint parity as sufficient.

7. Add a Poincare adaptive comparison mode.
   - [ ] Add a separate Diffrax-backed adaptive mode to
     `poincare_surfaces_jax_default.py` without changing the default fixed-step
     RK4 mode.
   - [ ] Emit distinct metadata for the Diffrax mode, including solver name,
     tolerance, `max_steps`, save/event strategy, and whether fixed quarter
     planes were reached by saved times or dense interpolation.
   - [ ] Produce side-by-side metrics against the current fixed-step RK4 path,
     with no overwrite of existing default JAX artifacts.
   - [ ] Keep the fixed-step RK4 path as the baseline until adaptive-mode
     artifacts are reviewed.

8. Evaluate `magnetic_axis_helpers.py` as a secondary cleanup.
   - [ ] Prototype a Diffrax tangent-map solve behind a separate helper, not as
     a replacement for `on_axis_iota_rk`.
   - [ ] Compare iota, `steps_taken`, and success status against the current
     hand-written Dormand-Prince loop and the SciPy RK45 oracle lane.
   - [ ] Decide whether reverse-mode support is a real requirement for this
     kernel. If it is not, do not add complexity solely to enable it.
   - [ ] Promote this path only if it reduces duplicated integrator code without
     weakening the derivative-heavy parity contract.

9. Promotion and cleanup.
   - [ ] Keep both backends until fieldline, particle, Poincare, and iota
     decisions are documented.
   - [ ] Promote Diffrax only for surfaces where it passes parity, AD,
     strict-transfer, compile, and memory gates.
   - [ ] If Diffrax is promoted, remove duplicated local solver code only where
     the old code no longer owns event/status semantics.
   - [ ] Update docs and parity manifests to distinguish reference backend,
     experimental backend, and promoted backend.

## Validation Plan

- [ ] Static dependency check:
  ```sh
  rg -n "diffrax|Diffrax" pyproject.toml requirements.txt docs/requirements.txt src tests examples docs \
    --glob '!docs/diffrax_tracing_integration_implementation_plan.md'
  ```
- [ ] Plan-only whitespace check that also works while this plan is untracked:
  ```sh
  awk '/[ \t]$/ { print FILENAME ":" FNR ": trailing whitespace"; bad=1 } END { exit bad }' \
    docs/diffrax_tracing_integration_implementation_plan.md
  ```
- [ ] Official Diffrax docs refresh before implementation:
  ```sh
  npx ctx7@latest library Diffrax \
    "docs/diffrax_tracing_integration_implementation_plan.md implementation review"
  npx ctx7@latest docs /patrick-kidger/diffrax \
    "docs/diffrax_tracing_integration_implementation_plan.md implementation review"
  ```
- [ ] Existing tracing regression gate before any implementation:
  ```sh
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/Users/suhjungdae/code/columbia/simopt-jax-clean-local/src:/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/lib/python3.13/site-packages \
  JAX_ENABLE_X64=1 \
    /Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/bin/python3.13 -S \
    -m pytest tests/jax/core/test_tracing_jax_item14.py
  ```
- [ ] Existing magnetic-axis regression gate before any implementation:
  ```sh
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/Users/suhjungdae/code/columbia/simopt-jax-clean-local/src:/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/lib/python3.13/site-packages \
  JAX_ENABLE_X64=1 \
    /Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/bin/python3.13 -S \
    -m pytest tests/field/test_magnetic_axis_helpers_jax_item21.py
  ```
- [ ] Existing Poincare regression gate before any implementation:
  ```sh
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/Users/suhjungdae/code/columbia/simopt-jax-clean-local/src:/Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/lib/python3.13/site-packages \
  JAX_ENABLE_X64=1 \
    /Users/suhjungdae/code/columbia/simsopt-jax/.miniforge/bin/python3.13 -S \
    -m pytest tests/geo/test_poincare_jax_default.py
  ```
- [ ] Diffrax fieldline prototype gate: endpoint, status, padded trajectory,
  mask, event payload, and `jax.grad` parity against the current solver on
  analytic fields.
- [ ] Known-solution numerical spot check: for a constant Cartesian magnetic
  field, the no-event fieldline solve must satisfy `x(t) = x0 + t * B` within
  the relevant `event_time_tracing` tolerance lane before it is compared only
  against the current solver.
- [ ] Diffrax event gate: first-event, earliest-stopping, levelset, and
  `phi_hits_count > max_phi_hits` overflow behavior match the current contract.
- [ ] Particle tracing gate: guiding-center and full-orbit tests still assert
  scan-compatible compiled behavior and physical invariants.
- [ ] Strict-transfer gate: run compiled Diffrax kernels under the repo's
  transfer guard to reject hidden host callbacks or broad host/device copies.
- [ ] Compile/memory gate: record compile time, HLO size or equivalent
  diagnostic, peak memory, and runtime for representative fieldline and
  particle traces against the current solver.
- [ ] Dependency gate: after adding Diffrax to dependencies, verify install,
  import, and locked resolution in the same environment used for JAX tests.

## Risks and Mitigations

- Risk: Diffrax changes event ordering, root localization, or event row payloads.
  Mitigation: Keep current event localization as the oracle and add explicit
  event-row parity tests before routing any public adapter call to Diffrax.

- Risk: Diffrax introduces top-level `while` behavior that weakens current
  reverse-mode AD tests for fieldline tracing.
  Mitigation: Require an explicit adjoint policy and keep the existing
  `scan`/reverse-mode tests as promotion gates.

- Risk: Saved Diffrax solution shapes do not map cleanly to the current padded
  trajectory and mask contract.
  Mitigation: Adapt the Diffrax result behind a private helper and reject any
  design that changes public result dataclass shapes in the first milestone.

- Risk: New dependency surface increases maintenance and environment burden.
  Mitigation: Keep Diffrax optional until dependency review, user approval, and
  locked validation pass.

- Risk: Poincare adaptive mode silently changes artifact semantics.
  Mitigation: Keep fixed-step RK4 as the default and write distinct Diffrax
  output metadata and filenames.

- Risk: Optimizer or geometry code gets pulled into the Diffrax migration.
  Mitigation: Keep the plan scoped to ODE tracing surfaces and explicitly mark
  optimizers, root solvers, and algebraic geometry kernels out of scope.

## Completion Criteria

- [ ] A dependency review records whether Diffrax is approved, rejected, or
  deferred.
- [ ] A private Diffrax fieldline prototype exists and is disabled by default.
- [ ] Existing tracing tests pass unchanged with the hand-written backend.
- [ ] Diffrax fieldline tests prove endpoint, event, status, shape, and
  reverse-mode gradient parity on analytic and representative fields.
- [ ] Public adapter routing remains unchanged until core parity passes.
- [ ] Poincare adaptive-mode artifacts are separate from fixed-step RK4
  artifacts and record accurate solver metadata.
- [ ] `magnetic_axis_helpers.py` has either a validated Diffrax prototype or a
  documented decision to keep the local solver.
- [ ] Documentation identifies which backend is reference, experimental, or
  promoted for each tracing surface.

## Open Questions

- Should Diffrax be allowed into the existing `JAX` optional dependency group,
  or should it be isolated behind a separate extra until promoted?
- Which Diffrax adjoint policy is acceptable for `trace_fieldline` reverse-mode
  AD and higher-order differentiation tests?
- Can Diffrax event handling reproduce the current event-row schema directly,
  or should Diffrax only supply dense output while current event localization
  remains repo-owned?
- Is exact `steps_taken` parity required for `magnetic_axis_helpers.py`, or is
  iota/value parity under the derivative-heavy tolerance enough?
- What compile-time and memory regression thresholds are acceptable before a
  Diffrax backend can be enabled in production traces?
