# Diffrax Field-Line Tracing Integration — Implementation Plan

> Created 2026-06-25. Repo: `simsopt-pr-jax-port-clean` (branch `pr/jax-port-clean`).
> Scope decision: **Option B — additive, parity-gated diffrax backend** (see Rationale). Not a replacement of the hand-rolled tracer.
> **Status 2026-06-25: IMPLEMENTED + reviewed (code-review-fix-loop + 4-lens Crucible PASS).** Backend `src/simsopt_jax/core/tracing_diffrax.py`, `integrator` knob + dispatch, adapter `integrator=`/`max_steps=`/`max_phi_hits=` params, `diffrax==0.7.2` extras, tests `tests/field/test_tracing_diffrax_fieldline.py` (28 pass). **Resolved open questions:** φ-localization uses `SaveAt(steps=True)` + the native `dopri5_step` sub-step (NOT `dense=True`); event root-finder = `None` (Bisection raises under vmap, VeryChord diverges → step-granularity stop, matching native). Fixed bugs: Event `direction=False`, status via `event_occurred`, `idx<0` stop row recorded, int32 dtype, seed-already-fired stop, firing-index strict `<=0`. Native path byte-unchanged + diffrax-free. GPU lane deferred to CI.

## Purpose

Wire [diffrax](https://github.com/patrick-kidger/diffrax) (Patrick Kidger's JAX ODE library) into this repo as a **vetted, optional integrator backend** for field-line tracing, selectable behind the *existing* `compute_fieldlines` / `FieldlineTracingResult` contract.

This is motivated by a concrete, observed defect: the surrogate's fixed-step field-line tracer renders fuzzy Poincaré plots (punctures scatter off the flux surfaces), while the C++ reference (`compute_fieldlines`, adaptive RK45 + high-order event localization) renders crisp nested surfaces. Adaptive Dopri5 + accurate crossing localization is the fix; diffrax is the standard, maintained implementation of exactly that.

## Goals

- A `trace_fieldline_diffrax(spec, y0, magnetic_field_fn, phis, stopping_criteria) -> FieldlineTracingResult` that is a **drop-in alternative** to `trace_fieldline` (`src/simsopt_jax/core/tracing.py:1212`), producing the byte-compatible `FieldlineTracingResult` payload.
- Backend selection via a typed `integrator` field on `FieldlineTracingSpec` (choices `{"dopri5_native", "dopri5_diffrax"}`, default `"dopri5_native"` — existing behavior unchanged), plumbed through `compute_fieldlines(..., integrator=...)` in `src/simsopt_jax_adapters/field/tracing.py:650`.
- φ-plane crossing recording and the **geometric** stopping criteria (`MinR/MaxR/MinZ/MaxZ/Levelset`) supported on the diffrax path — the Poincaré-complete set.
- A batched, `vmap`-able driver `trace_fieldlines_diffrax_batched` matching `trace_fieldlines_batched` (`src/simsopt_jax/core/tracing.py`).
- `diffrax` added to the `JAX` and `JAX_GPU` extras in `pyproject.toml`.
- Parity gates: diffrax backend agrees with (a) the hand-rolled `dopri5_native` tracer and (b) the C++ analytic-field references, to tolerance.

## Non-Goals (first cut)

- Guiding-center (Cartesian + Boozer) and full-orbit diffrax backends. The hand-rolled paths stay. (Future follow-up.)
- `IterStoppingCriterion` / `ToroidalTransitStoppingCriterion` on the diffrax path — these need integrator/step state that diffrax's stateless `Event` cond-fn doesn't see cleanly. Handle via `max_steps` (Iter) and post-solve truncation (Transit) later; not required for Poincaré.
- **Replacing** the hand-rolled tracer (Option A). That is a separate, larger decision; this plan keeps both behind one contract and produces the A/B parity data that would justify A later.
- `requirements.txt` changes — it is a legacy minimum-bounds file (`jax >= 0.2.5`), not used by CI or production. Dependency truth lives in `pyproject.toml` extras.

## Current Context

The repo **already has** an in-repo adaptive Dopri5 tracer — diffrax's hand-rolled twin. Key surfaces (verified):

- `src/simsopt_jax/core/tracing.py` (4358 lines):
  - `dopri5_step` (`:778`) — the exact Dormand-Prince RK4(5) Butcher tableau (same method as diffrax `Dopri5`).
  - `_dopri5_adaptive_step` (`:1030`) + PI(0.2) controller constants (`_DOPRI5_EXP=0.2`, `_SAFETY=0.9`, `_MIN_FACTOR=0.2`, `_MAX_FACTOR=5.0`, `:164`).
  - `trace_fieldline` (`:1212`) — adaptive driver in a fixed-length `lax.scan`; interleaves per-step φ-crossing detection and stopping-criteria checks.
  - `state_at_fraction` (`:1405`) — **5th-order** sub-step crossing localizer (re-runs a `dopri5_step` from the prior accepted state; NOT a linear interpolant — this is what makes punctures land *on* the surface).
  - `bracket_root_jax` (`:899`) — Illinois false-position event localizer.
  - `_scan_angle_plane_events` (`:1090`), `_apply_stopping_criteria_events` (`:1160`) — reusable crossing/stopping recorders.
  - `FieldlineTracingSpec` (`:374`, registered pytree), `FieldlineTracingResult` (`:424`, registered pytree).
  - Stopping-criterion dataclasses (`:479`–`:633`): `MinR/MaxR/MinZ/MaxZ/ToroidalTransit/Iter/Min-MaxToroidalFlux/Levelset`.
  - `trace_fieldlines_batched` — `vmap` over lanes.
- `src/simsopt_jax_adapters/field/tracing.py` (856 lines):
  - `compute_fieldlines(field, R0, Z0, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None)` (`:650`) → `_compute_fieldlines_jax` (`:785`) — **live**, returns `(res_tys, res_phi_hits)` matching the C++ column layout.
  - `_translate_stopping_criteria_to_jax` (`:674`) — maps C++ criterion instances → JAX dataclasses.
  - Field requirement: `field.jax_B_at(point) -> B[3]` (`_require_jax_field_B`, `:186`).
- Upstream parity target `src/simsopt/field/tracing.py:661` — `compute_fieldlines(field, R0, Z0, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None)` → `(res_tys, res_phi_hits)`; `res_phi_hits` columns `[t, idx, x, y, z]`, `idx>=0` = φ-plane `phis[idx]`, `idx<0` = stopping criterion `-1-idx`.
- Packaging `pyproject.toml`: `JAX` extra (`:72`) pins `jax==0.10.0`, `equinox>=0.11.11`, `lineax>=0.1.1`, `optimistix>=0.1`; `JAX_GPU` extra (`:85`) mirrors with CUDA. No lockfile (immutable resolution = exact pins). pytest markers incl. `slow`, `jax_gpu_pure`, `jax_contract` (`:161`).
- Tests: `tests/field/test_tracing_jax_item16.py`, `_item16_extended.py` (JAX `compute_fieldlines` end-to-end), `tests/field/test_fieldline.py` (C++ analytic refs: `test_poincare_toroidal`, `test_poincare_tokamak`), `tests/jax/core/test_tracing_jax_*.py`. Parity-test convention: `jax.config.update("jax_enable_x64", True)` first; `repo_bootstrap.bootstrap_local_simsopt`; `conftest.parity_mode_case(...)` for CPU/GPU skip-guarded parametrization.
- **Dependency check (done 2026-06-25):** `diffrax==0.7.2` requires `python>=3.11`, `jax>=0.4.38`, `equinox>=0.11.10`, `lineax>=0.0.5`, `optimistix>=0.1.0` (+ small new transitive: `jaxtyping`, `wadler-lindig`, `typing-extensions`). **All satisfied** by the repo's pins; no conflict.
- **diffrax API** (v0.7.2, confirmed via ctx7): `diffeqsolve(term, solver, t0, t1, dt0, y0, saveat=, stepsize_controller=, event=, adjoint=, max_steps=)`; `ODETerm(lambda t, y, args: ...)`; `Dopri5()`; `PIDController(rtol, atol, pcoeff, icoeff, dcoeff, dtmax, ...)`; `SaveAt(t0=, t1=, ts=, steps=, dense=, fn=)`; `Event(cond_fn, root_finder=)`.

## Rationale

**Why diffrax at all, given the hand-rolled tracer works?** Two reasons: (1) the surrogate's *fixed-step* tracer is visibly inferior (fuzzy Poincaré), and adaptive+high-order-crossing is the fix; (2) for an upstream PR, a 4358-line hand-rolled RK45 + custom event localizer is a large review/maintenance surface — a vetted dependency is a credible long-term simplification.

**Why Option B (additive) over Option A (replace)?**
- The word in the request — "**integration** plan" — and this repo's established **multi-backend idiom** (`src/simsopt_jax/solve/{scipy,optax,optimistix,simsopt}` behind contracts) both point to an additive backend.
- B is additive and reversible: existing behavior and every passing tracing test are untouched (default stays `dopri5_native`).
- B *produces the evidence* (A/B parity, perf) that an eventual A would need. A without that evidence is a blind risky refactor.
- The diffrax backend module is built either way; A would merely additionally delete the old core. No work is wasted if the user later chooses A.

**Why this is SSOT-safe:** the contract (`FieldlineTracingResult`, `compute_fieldlines` output columns) remains the single source of truth; both backends conform to it. The `integrator` knob is **externally-owned behavior config** (engine/fidelity choice owned by the caller) — allowed under the design rules when typed, documented, and tested.

## Assumptions

- diffrax `SaveAt(steps=True)` (optionally `+ dense=True`) yields fixed-shape `(max_steps,)` step grids paddable to the `FieldlineTracingResult` layout, and is `vmap`/`jit`-compatible with a static `max_steps`. (Confirm exact `steps`+`dense` combinability in 0.7.2 during impl; fallback in Risks.)
- An adaptive Dopri5 solution with tight `rtol/atol` agrees with the hand-rolled `dopri5_native` solution to **solution tolerance** (not bit-identity — two different adaptive step sequences cannot be bit-identical). Parity gate is `assert_allclose` at the tolerance the lane already uses for tracing, not byte-equality.
- The geometric stopping criteria (`MinR/MaxR/MinZ/MaxZ/Levelset`) are pure functions of the instantaneous state `y` → clean diffrax `Event` cond-fns. (True by construction.)
- The `magnetic_field_fn` passed to the backend is JAX-traceable (already required by the existing path).

## Implementation Plan

### Phase 0 — Baseline & dependency (de-risk first)
- [ ] Confirm the existing `dopri5_native` `compute_fieldlines` runs and matches C++ on the `test_poincare_tokamak` analytic field (closed-form circular orbit) — establishes the parity baseline the diffrax backend must hit. (This is also background task #1 on the real clean2p5mm field.)
- [ ] Add `diffrax==0.7.2` to the `JAX` and `JAX_GPU` extras in `pyproject.toml` (next to `equinox`/`lineax`/`optimistix`). Match the sibling `>=`/ceiling style. **Flag as a new direct dependency for user ack.**
- [ ] Verify `pip install -e ".[JAX]"` resolves with diffrax in a scratch env (we already dry-ran the metadata; do a real resolve once).

### Phase 1 — Core diffrax fieldline backend (new file; sole owner)
New module `src/simsopt_jax/core/tracing_diffrax.py` (keeps `tracing.py` untouched except the dispatch hook in Phase 2):
- [ ] `_fieldline_term(magnetic_field_fn)` → `diffrax.ODETerm(lambda t, y, args: magnetic_field_fn(y))` (mirrors `fieldline_rhs`, `:749`).
- [ ] `_fieldline_controller(spec)` → `diffrax.PIDController(rtol=spec.rtol, atol=spec.atol, dtmax=spec.dtmax, ...)`, configured toward the hand-rolled elementary I-controller (pcoeff=0, icoeff≈1, dcoeff=0; same safety/clip semantics as far as diffrax exposes). Document that exact-controller-match is not the parity contract.
- [ ] `_fieldline_events(stopping_criteria)` → `diffrax.Event(cond_fn)` where `cond_fn(t, y, args, **kw)` returns a scalar that goes ≤0 when **any** supported geometric criterion fires (min over criteria). Reject unsupported criteria (`Iter`/`Transit`/flux) with a clear `NotImplementedError` on the diffrax path (loud, per integrity-boundary rules — not silently ignored).
- [ ] `trace_fieldline_diffrax(spec, y0, magnetic_field_fn, phis, stopping_criteria) -> FieldlineTracingResult`:
  - solve `diffeqsolve(term, Dopri5(), t0=0, t1=spec.tmax, dt0=<initial step>, y0, saveat=SaveAt(t0=True, steps=True, dense=True), stepsize_controller=controller, event=event, max_steps=spec.max_steps)`.
  - **trajectory/mask/steps_taken/t_final/status**: map `sol.ts/sol.ys` (padded with `inf`) → `(max_steps+1,4)` `[t,x,y,z]` + finite-mask; derive `status` (0 normal, 1 max-steps, `-1-i` from the fired criterion identified at the terminal state).
  - **phi_hits/phi_hits_count**: reuse the existing `_scan_angle_plane_events` (`:1090`) + `bracket_root_jax` (`:899`) over consecutive saved steps; for the sub-step state use **diffrax dense output** `sol.evaluate(t)` (5th-order) — the crisp-puncture path. Produces the identical `(max_phi_hits,5)` `[t,idx,x,y,z]` layout.
  - **Interface comment** (≤5 lines) drafted before the body, re-checked after (two-phase rule).
- [ ] `trace_fieldlines_diffrax_batched(...)` → `jax.vmap` over lanes, matching `trace_fieldlines_batched`'s signature/return.

### Phase 2 — Selection plumbing (owns edits to `tracing.py` + adapter)
- [ ] Add `integrator: str = "dopri5_native"` as a **static meta field** on `FieldlineTracingSpec` (`:374`) and its `register_dataclass` `meta_fields` (`:417`). Document the two choices.
- [ ] In `trace_fieldline` / `trace_fieldlines_batched`, dispatch on `spec.integrator` to the native path (default) or `tracing_diffrax`. Keep the native path the literal default so all existing callers/tests are unchanged.
- [ ] In the adapter `compute_fieldlines` (`:650`), add `integrator="dopri5_native"` param, thread it into the spec. (Tier 3 surface — see API gate below: additive optional param, default preserves behavior.)

### Phase 3 — Tests (owns new test files)
- [ ] `tests/field/test_tracing_diffrax_fieldline.py`:
  - **C++/analytic parity:** diffrax backend vs the closed-form `test_poincare_tokamak`/`test_poincare_toroidal` orbits (non-tautological reference).
  - **Backend A/B parity:** `dopri5_diffrax` vs `dopri5_native` on a non-trivial field (e.g. the item16 fixture) — trajectory endpoint + φ_hits `assert_allclose` to the tracing lane tolerance.
  - **Stopping-criteria parity:** `MaxR/MinR/MaxZ/MinZ/Levelset` each fire at the same crossing (status + hit row) on both backends.
  - **vmap batch parity:** batched diffrax == per-lane diffrax == native, plus a GPU-skip-guarded lane via `parity_mode_case`.
  - **Unsupported-criterion guard:** `Iter`/`Transit` on the diffrax path raises `NotImplementedError` (loud).
- [ ] Follow the repo convention: x64 first, `bootstrap_local_simsopt`, `parity_mode_case`, state assertions (not mock/interaction). DAMP over DRY.

### Phase 4 — Review loop & docs
- [ ] Run targeted tests, then the repo's required checks (ruff, the tracing test subset).
- [ ] `crucible` adversarial review until strict PASS (per `/requirements-e2e-review-loop`).
- [ ] Update this plan's checkboxes; note the dependency in the PR description.

## Validation Plan

- [ ] `dopri5_diffrax` reproduces the closed-form tokamak Poincaré orbit (radius invariant) to tracing tolerance.
- [ ] `dopri5_diffrax` vs `dopri5_native`: trajectory endpoint and `res_phi_hits` agree via `np.testing.assert_allclose` at the lane tolerance; **Poincaré render is visibly crisp** (the actual acceptance — compare against `.m18-adjoint-artifacts/clean2p5mm_poincare/PoincarePlot_clean2p5mm_default.png`).
- [ ] Each geometric stopping criterion: identical `status` and terminal hit row across backends.
- [ ] Batched == per-lane == native (vmap correctness).
- [ ] `pip install -e ".[JAX]"` and `".[JAX_GPU]"` resolve with diffrax.
- [ ] `ruff` clean on new files; no regression in the existing `tests/field/test_tracing_jax_item16*.py` (native path untouched).
- [ ] crucible strict PASS.

## Risks and Mitigations

- **Risk:** `SaveAt(steps=True)` + `dense=True` not combinable, or dense output not `vmap`-safe in 0.7.2.
  **Mitigation:** Fallback to `SaveAt(steps=True)` only, and localize crossings by re-running the existing `dopri5_step` sub-step (the `state_at_fraction` approach, `:1405`) instead of `sol.evaluate`. Both give 5th-order crossings. Confirm the API against diffrax 0.7.2 docs (ctx7) before committing the approach.
- **Risk:** diffrax `Event` termination interacts badly with fixed `max_steps` / produces a different `status` taxonomy.
  **Mitigation:** Identify the fired criterion by evaluating each geometric predicate at the terminal `sol.ys[-1]`; unit-test the status mapping directly.
- **Risk:** PIDController cannot be configured to exactly match PI(0.2), so A/B trajectories differ more than expected.
  **Mitigation:** Parity gate is solution-agreement-to-tolerance, not bit-identity. Tighten `rtol/atol` in the A/B test so both converge to the same ODE solution; if divergence exceeds tolerance, investigate the field/RHS, not the controller.
- **Risk:** new transitive deps (`jaxtyping`, `wadler-lindig`) drift.
  **Mitigation:** They are diffrax-pinned and pure-python; no separate pin needed. Note in PR.
- **Risk:** scope creep into GC/Boozer/full-orbit.
  **Mitigation:** Hard non-goal for this cut; the dispatch only routes the fieldline path to diffrax.

## API Evolution Gate (adapter `compute_fieldlines` is Tier 3)

- **Observable delta:** new optional `integrator=` param; default `"dopri5_native"` preserves *all* current observable behavior (same trajectories, same `res_phi_hits`). No timing/ordering/error-message change on the default path.
- **Caller inventory:** `compute_fieldlines` callers in `tests/field/test_tracing_jax_item16*.py`, the adapter's own routing, and any example harnesses. All unaffected (default).
- **Migration:** none required; opt-in only via `integrator="dopri5_diffrax"`.
- **Compatibility test:** existing `test_tracing_jax_item16*` must pass unchanged (proves the default path is untouched).
- **Rollback:** remove the diffrax route + revert the extras pin; the native path is independent.

## Completion Criteria

- [ ] `trace_fieldline_diffrax` + batched driver land in `src/simsopt_jax/core/tracing_diffrax.py`, conforming to `FieldlineTracingResult`.
- [ ] `integrator` knob plumbed through spec + adapter; default unchanged.
- [ ] All Validation Plan items green, incl. a **crisp** clean2p5mm Poincaré render via the diffrax backend.
- [ ] `diffrax==0.7.2` in `JAX`/`JAX_GPU` extras; user-acknowledged.
- [ ] crucible strict PASS; plan checkboxes updated.

## Open Questions

- **A vs B:** proceeding with **B** (additive). If the user wants **A** (replace the hand-rolled core for PR-cleanliness), the backend is the same; A additionally deletes `dopri5_step`/`_dopri5_adaptive_step`/the native `trace_fieldline` body and re-validates every tracing test against diffrax. Decision can be deferred until the A/B parity data exists.
- **Deferred criteria:** confirm `Iter`/`ToroidalTransit` on the diffrax path are genuinely not needed near-term (Poincaré uses only box guards). If needed, design the stateful-event handling as a follow-up.
- **Exact diffrax `SaveAt` strategy** (steps+dense vs steps+sub-step) — resolved in Phase 1 against the 0.7.2 API.
