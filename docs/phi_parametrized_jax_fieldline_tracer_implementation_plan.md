# φ-Parametrized JAX Field-Line Tracer — Implementation Plan

> Created: 2026-06-26 · Status: PLAN (not started) · Author: orchestration session, repo `simsopt-pr-jax-port-clean` @ `pr/jax-port-clean` (HEAD `f66e641ff`)

## Purpose

Add a **differentiable, φ-parametrized** field-line tracer to the JAX port. It exists to
be the clean, reverse-mode-differentiable substrate for **in-the-loop confinement
objectives** (Poincaré return maps / island width), where the existing arc-length tracer is
a poor fit because its Poincaré output depends on per-step crossing detection and a step
budget.

This is **additive**. The C++ tracer remains the general/diagnostic tracer; the arc-length
JAX tracer (`dopri5_native` + the new `dopri5_diffrax` backend) is untouched. See
`docs/diffrax_tracing_integration_implementation_plan.md` for that work.

## Goals

- A standalone JAX function that integrates `dR/dφ, dZ/dφ` in the toroidal angle φ and
  returns `(R, Z)` at exactly the requested `phis` — no crossing detection, no step-budget
  density wall, static output shape `(n_phi,)` per line.
- **Reverse-mode differentiable** end to end (`jax.grad` / `jax.jacrev` through the solve
  via `diffrax.RecursiveCheckpointAdjoint`), with a gradient correctness test. This is the
  entire justification for the tracer; it is a first-class acceptance gate, not a nicety.
- `vmap`-batched + optionally `shard_map`-sharded over many seed lines, mirroring the
  existing batched fieldline path.
- A public adapter (`compute_poincare_phi`) that accepts the **same** JAX field objects
  `compute_fieldlines` already accepts.
- Honest, loud behavior in the φ-param singular regime (B_φ → 0): NaN / `escaped`, never a
  silently-wrong puncture.

## Non-Goals

- **No** replacement of the C++ tracer or the arc-length JAX tracer. Both stay.
- **No** shoehorning φ-param output into `FieldlineTracingResult` (arc-length-shaped). New
  thin result type instead — see Rationale.
- **No** general-domain validity. φ-param is singular at B_φ = 0; this tracer is scoped to
  the B_φ-dominated nested-surface regime (the confinement-objective regime). Separatrix /
  X-point / reversed-B_φ tracing stays on the arc-length / C++ path.
- **No** new in-loop *physics objective* shipped as production here. Phase 4 ships a
  minimal differentiable demonstrator + wiring notes; promoting it into the optimizer's
  confinement gate is a separate decision.
- **No** guiding-centre / full-orbit / Boozer φ-param variants. Field-line only.

## Current Context

Grounded in the repo as of HEAD `f66e641ff`:

- **Arc-length contract** (`src/simsopt_jax/core/tracing.py`):
  - `FieldlineTracingSpec` (`:374`) is time/arc-length: `tmax`, `max_steps` (static, the
    `lax.scan` length), per-lane `dtmax`, `integrator: Literal["dopri5_native","dopri5_diffrax"]`
    (`:421`).
  - `FieldlineTracingResult` (`:431`) columns are `(t,x,y,z)` with a live-prefix `mask`,
    `steps_taken`, `status`, and a crossing-detection `phi_hits` buffer `[t,idx,x,y,z]`.
    **Every field assumes arc-length semantics + step-grid + crossing localization.**
  - `trace_fieldline` (`:1219`) dispatches `dopri5_diffrax` lazily (`:1263`); native path is
    a fixed-length `jax.lax.scan` (`:1361` cond / `:1387` body) doing per-step φ-crossing
    detection via `_scan_angle_plane_events` (`:1097`) + `bracket_root_jax` (`:906`).
  - Batched path: `_make_fieldline_trace_one` (`:1610`) → `trace_fieldlines_batched` (`:1657`)
    `vmap`s over lanes with **per-lane `dtmax`**, and `shard_map`s for multi-device (`:1700`).
- **Adapter** (`src/simsopt_jax_adapters/field/tracing.py`):
  - `compute_fieldlines` (`:650`) → `_compute_fieldlines_jax` (`:804`).
  - Field is resolved via `_resolve_jax_field_B` (`:197`): prefers `jax_B_at_state` +
    `jax_tracing_state`, else `jax_B_at` (Cartesian B at a **Cartesian** point). Repo search
    finds `jax_B_at` on the analytic/interpolated field wrappers, and no `jax_B_at` on
    `BiotSavartJAX` itself; direct `BiotSavartJAX` support therefore requires the same
    wrapper/state route that `compute_fieldlines` already requires. The new tracer reuses
    this resolver and should not add a second field-interface source of truth.
  - Output to the gallery is host lists: `res_tys` (live `(t,x,y,z)` per line) and
    `res_phi_hits` (`[t,idx,x,y,z]` per line) via `_batched_jax_trace_payloads` (`:106`).
- **φ-param equation contract**:
  - For cylindrical field components `(B_R, B_φ, B_Z)`, a 2-state solver whose Diffrax
    time is already φ integrates the quotient derivatives `dR/dφ = R*B_R/B_φ` and
    `dZ/dφ = R*B_Z/B_φ`.
  - **Division by `B_φ` is the singularity.** A `sign(B_φ)` factor belongs only to a
    separate pseudo-time system that carries φ as a state; it must not be multiplied into
    the 2-state `d/dφ` quotient.

## Rationale

**Why φ-param at all.** For a differentiable Poincaré objective, arc-length is the wrong
parametrization: punctures come from per-step crossing detection (`bracket_root_jax`,
`_scan_angle_plane_events`) and density is bounded by `max_steps`. φ-param integrates *in*
the toroidal angle, so `SaveAt(ts=phis)` returns punctures at exactly the planes you ask
for — no root-finds, static shape, and the entire `Event` root-finder fragility we hit on
the arc-length diffrax backend (Bisection raises under vmap; VeryChord diverges; settled on
`root_finder=None`) simply does not arise.

**Why a new result type, not a third `integrator=` enum (the design-it-twice).**
- *Option A — `integrator="fieldline_phi_diffrax"` returning `FieldlineTracingResult`.*
  Rejected. The φ-param output has no `t` axis (φ is the independent variable), no per-step
  trajectory or live `mask`, and no crossing `phi_hits` (the punctures *are* the output).
  Every field of `FieldlineTracingResult` would be a lie or a stub. The `integrator` enum
  on `FieldlineTracingSpec` carries a contract promise ("byte-compatible
  `FieldlineTracingResult`"); a φ-param branch breaks it. This is textbook information
  leakage — the arc-length representation decision bleeding into a module that does not
  share it (SOFTWARE_DESIGN: "different layer, different abstraction"; "specific types beat
  generic containers").
- *Option B — standalone `trace_poincare_phi` + thin `PoincareReturnResult`.* Chosen.
  Honest contract, φ-grid-shaped, zero changes to the arc-length contract. Output is
  `(R,Z)` at `phis` plus an `escaped` mask and a repo-local normalized `status`.

**Why no denominator epsilon.** Adding `1/(bp+eps)` would convert a loud, correct failure
(NaN where the model is invalid) into a silently-wrong puncture — a defensive fallback the
guardrails forbid and a correctness landmine for an objective. The implementation uses the
singular quotient directly and does not add a pseudo-time `sign(B_φ)` factor inside the
2-state `d/dφ` RHS. B_φ→0 must fail loud, guarded by the bounding-box `Event`,
nonfinite-output handling, non-success status, and an explicit test. The domain restriction
is documented, not papered over.

## Assumptions

- The confinement regime of interest has B_φ bounded away from 0 on the traced surfaces
  (true for stellarator/tokamak core flux surfaces). Validated by the singular-regime test
  asserting loud failure outside it.
- The JAX field objects expose Cartesian `jax_B_at` / `jax_B_at_state` (via
  `_resolve_jax_field_B`); a cylindrical-B wrapper (≤30 LOC coordinate rotation) bridges to
  the φ-param RHS. No new field method is required on the field classes, and direct
  `BiotSavartJAX` is not accepted unless it comes through an existing resolver-compatible
  wrapper/state object.
- `diffrax==0.7.2` is available (already pinned in `pyproject.toml` JAX/JAX_GPU extras from
  the arc-length diffrax work). `RecursiveCheckpointAdjoint`, `SaveAt(ts=...)`, `Event`,
  `PIDController(dtmin=...)`, `throw=False`, and `max_steps` exist in 0.7.2 (confirmed by
  ctx7 + the sibling JAX env signatures).
- `phis` is a 1-D monotonic, unwrapped φ grid. The public adapter owns host-side validation
  and derives any default `max_steps` from a host `np.asarray(phis)`; the jitted core
  receives concrete static sizes.
- x64 is enabled for tracing (repo default); punctures are float64.

## Implementation Plan

### Phase 0 — Contract design (Tier 2: new module boundary; design-it-twice done above)
- [ ] New module `src/simsopt_jax/core/tracing_poincare_phi.py`. Draft the interface
      comment for `PoincareReturnResult` and `trace_poincare_phi` in ≤5 lines **before**
      implementing; if it won't fit, the abstraction is wrong — stop and redesign.
- [ ] `PoincareTracingSpec` (frozen dataclass, `register_dataclass`): `rtol`, `atol`,
      `min_step_size` (data); `max_steps: int` (meta/static, always concrete in the core);
      `bounds_R`, `bounds_Z` (data, default `(0, inf)` / `(-inf, inf)`). The adapter may
      accept `max_steps=None`, but it must derive `int(abs((phis[-1]-phis[0]))*1000)` on
      the host before constructing the spec. No `tmax`, no per-lane `dtmax` (φ-span
      replaces `tmax`; PID + `dtmin` replace `dtmax`).
- [ ] `PoincareReturnResult` (frozen dataclass, `register_dataclass`): `punctures`
      `(n_phi, 2)` float64 `[R, Z]` per line; `escaped` `(n_phi,)` bool (True where a save
      point is NaN/nonfinite, i.e. the line left the box or the solve failed before that
      plane); `status` int32 normalized from Diffrax (`0` successful, `-1` box event,
      `1` other non-success such as max-step exhaustion/singular failure). Do not expose
      `sol.result` as an `int32`: in diffrax 0.7.2 it is an Equinox enumeration item, not a
      scalar array leaf. Decide whether to also carry Cartesian `xyz` punctures for reuse
      by the existing surface classifier (only if a consumer needs it — YAGNI otherwise).

### Phase 1 — Core single-line φ-param solve
- [ ] `_cyl_B_from_cartesian_field(field_fn)` → `rpz_B(R, phi, Z)`: build
      `xyz = [R cosφ, R sinφ, Z]`, call `field_fn(xyz)` (Cartesian `[Bx,By,Bz]`), rotate to
      `[B_R, B_φ, B_Z] = [Bx cosφ + By sinφ, -Bx sinφ + By cosφ, Bz]`. Pure, ≤30 LOC.
- [ ] `_phi_odefun(phi, rz, field_fn)`: `R = rz[0]`;
      `(B_R, B_φ, B_Z) = rpz_B(R, phi, rz[1])`; return `[R*B_R/B_φ, R*B_Z/B_φ]`
      (2-state `[R,Z]`; φ is the time, not a state). **No epsilon and no `sign(B_φ)`
      multiplier on the 2-state quotient.** The implementation comment should state the
      quotient and singular-domain contract directly.
- [ ] `trace_poincare_phi(spec, r0, z0, phis, field_fn)`:
      require monotonic `phis`; compute the signed step floor (`min_step_size` for
      increasing grids, `-abs(min_step_size)` for decreasing grids);
      call `diffrax.diffeqsolve(ODETerm(_phi_odefun), diffrax.Dopri5(), t0=phis[0],
      t1=phis[-1], dt0=signed_min_step, y0=[r0,z0], saveat=SaveAt(ts=phis),
      stepsize_controller=PIDController(rtol, atol, dtmin=signed_min_step),
      event=Event(box_cond_fn), adjoint=RecursiveCheckpointAdjoint(), max_steps,
      throw=False)`. Map `inf→nan`; `escaped = ~isfinite`; normalize
      Diffrax `sol.result` to `status`; assemble `PoincareReturnResult`.
- [ ] Box `cond_fn` = `(R<bounds_R[0])|(R>bounds_R[1])|(Z<bounds_Z[0])|(Z>bounds_Z[1])`
      with `root_finder=None` (the vmap-robust choice already validated on the arc-length
      backend).

### Phase 2 — Batched + sharded entry
- [ ] `trace_poincare_phi_batched(spec, r0s, z0s, phis, field_fn, magnetic_field_state=None)`:
      `jax.vmap` over `(r0, z0)` (and `field_state` via `in_axes=(0,0,None)` when present),
      mirroring `_trace_fieldlines_batched_unsharded` (`:1634`). No per-lane `dtmax`, so the
      signature is simpler than the fieldline path.
- [ ] Reuse `trajectory_batch_sharding_config` / `maybe_shard_trajectory_batch_inputs`
      (from `simsopt_jax.core.sharding`) for the multi-device `shard_map`, with `out_specs`
      sharded on the lane axis — same pattern as `trace_fieldlines_batched` (`:1683-1736`).

### Phase 3 — Public adapter
- [ ] `compute_poincare_phi(field, R0, Z0, phis, *, rtol=1e-8, atol=1e-8, max_steps=None,
      bounds_R=(0,inf), bounds_Z=(-inf,inf), comm=None)` in
      `src/simsopt_jax_adapters/field/tracing.py`. Reuse `_resolve_jax_field_B` (`:197`) +
      `_cyl_B_from_cartesian_field`; validate host `phis` as 1-D monotonic, derive concrete
      `max_steps` when `None`, and reject zero-span grids before JIT. Use
      `parallel_loop_bounds` for MPI split (match `_compute_fieldlines_jax`); call
      `trace_poincare_phi_batched`; device→host via `_jax_trace_host_array`.
- [ ] Output: φ-grid `(R, Z)` arrays of shape `(n_phi, nlines)` as the primary return
      (clean for differentiable objectives), plus a thin converter to the existing
      per-line puncture-list shape the Poincaré plotting consumes (so it can drop into the
      gallery for a visual cross-check). Decide primary vs convenience — see Open Q.

### Phase 4 — Differentiable demonstrator + wiring notes
- [ ] A minimal differentiable scalar through punctures (e.g. a return-map residual
      `‖(R,Z)|_{φ=2π} − (R,Z)|_{φ=0}‖` for a periodic seed, or an island-width proxy) with a
      `jax.grad` that returns finite, FD-matching gradients. Proves the AD path end to end.
- [ ] Doc note (in this file + the module docstring): how the existing confinement metrics
      (Greene residue, converse-KAM, WBA, multisurface-QS) could consume a differentiable
      φ-tracer as a shared integration substrate — without committing to that wiring here.

### Phase 5 — Tests + review loop
- [ ] `tests/field/test_tracing_poincare_phi.py` (see Validation Plan for the cases).
- [ ] `ruff` clean (`uvx ruff@0.15.15 check`).
- [ ] `crucible` adversarial review loop to strict PASS (per `requirements-e2e-review-loop`).

## Validation Plan

Dev env: use an interpreter with the repo's `JAX` extra installed. The known local one is
`/Users/suhjungdae/code/columbia/simsopt-jax-shared-jax/.conda/jax/bin/python` (py3.11,
jax 0.10.0, diffrax 0.7.2). The default `python` in this checkout currently lacks
`diffrax`, so validation commands should be explicit, e.g.
`JAX_PLATFORMS=cpu /Users/suhjungdae/code/columbia/simsopt-jax-shared-jax/.conda/jax/bin/python -m pytest tests/field/test_tracing_poincare_phi.py`.
`tests/conftest.py` already runs `repo_bootstrap.bootstrap_local_simsopt()` and forces x64;
direct scripts must still assert `simsopt_jax.__file__` is under this repo's `src/`.

- [ ] **Closed-surface invariant** — pure toroidal field (`B ∝ ê_φ/R`, B_φ ≠ 0 everywhere):
      every puncture returns to the seed `(R,Z)` each period; `escaped` all False. (Analytic
      ground truth; non-tautological.)
- [ ] **Cross-tracer parity** — on a nested-surface case where both are valid, compare
      φ-param punctures against the **arc-length native** tracer's φ-plane hits after
      aligning semantics: φ-param `phis` are unwrapped/monotonic, while native `phis` are
      target planes modulo `2π` and do not record the seed as a crossing. Exclude or
      separately check the seed plane, group native hits by `(transit, idx)`, and compare
      the matching `(R,Z)` punctures to integrator tolerance (~1e-7).
- [ ] **Differentiability gate** — `jax.grad`/`jacfwd` of the Phase-4 scalar w.r.t. a field
      DOF (or seed) is finite and matches central finite differences to ~1e-5. **Hard gate.**
- [ ] **Singular-regime loud failure** — a field/seed driven toward B_φ→0 yields NaN
      punctures + `escaped`/nonzero `status`, **not** a finite wrong number. Proves
      the no-epsilon decision fails safe.
- [ ] **φ-RHS quotient/sign regression** — analytic cylindrical field with known
      `dR/dφ = R*B_R/B_φ`, including a `B_φ < 0` case, catches any accidental
      pseudo-time `sign(B_φ)` multiplier in the 2-state RHS.
- [ ] **vmap == per-lane** — batched punctures equal looped single-line calls (atol ~1e-7,
      allowing vmap FP reassociation, as on the arc-length backend).
- [ ] **jit static shapes** — single + batched under `jax.jit` compile with static output
      shape `(n_phi, …)`.
- [ ] **Box exit** — a line crossing `bounds_R/Z` mid-trace marks all subsequent save
      points `escaped`, finite ones before it correct (inf→nan semantics).
- [ ] **Cylindrical wrapper unit test** — `_cyl_B_from_cartesian_field` rotation correct on
      a known analytic B (e.g. constant Cartesian B → correct R/φ components vs angle).

## Risks and Mitigations

- **Risk:** B_φ singularity produces garbage inside the valid-looking domain.
  **Mitigation:** singular-domain handling stays loud (no epsilon), with a box `Event`,
  explicit loud-failure test, and documented domain scope. Do not add defensive denominator
  guards.
- **Risk:** second tracer contract increases surface area / drift from the arc-length one.
  **Mitigation:** keep `PoincareReturnResult` thin and φ-grid-shaped; zero edits to
  `FieldlineTracingSpec`/`FieldlineTracingResult`; the two share the field resolver and
  sharding helpers (SSOT for those), nothing else.
- **Risk:** solver/tableau divergence from the arc-length backend.
  **Mitigation:** use `Dopri5` for repo consistency and cross-tracer parity. Do not expose a
  public `solver` knob for this internally-owned implementation choice.
- **Risk:** the differentiable objective is the whole point but has no production consumer.
  **Mitigation:** Phase 4 demonstrator + gradient gate proves the capability; framed as
  additive; promotion into the confinement gate is a separate, explicit decision.
- **Risk:** cylindrical conversion sign error silently biases punctures.
  **Mitigation:** dedicated rotation unit test + the closed-surface invariant catch it.

## Completion Criteria

- [ ] `trace_poincare_phi` + `trace_poincare_phi_batched` + `PoincareReturnResult` +
      `PoincareTracingSpec` implemented; arc-length contract untouched.
- [ ] `compute_poincare_phi` adapter accepts the same fields as `compute_fieldlines`.
- [ ] All Validation Plan checks green, **including the differentiability gate and the
      singular-regime loud-failure test**.
- [ ] `ruff` clean; `crucible` strict PASS.
- [ ] `diffrax` dependency already acknowledged (carried over from the arc-length diffrax
      work; no new direct dependency introduced by this plan).
- [ ] Module docstring + this doc state the domain scope (B_φ-dominated) and the
      additive-capability framing.

## Open Questions

- **Primary output shape:** φ-grid `(n_phi, nlines)` R,Z arrays vs the gallery per-line
  puncture list. Lean φ-grid primary (best for objectives) + a convenience converter.
- **First consumer:** does an existing confinement objective (residue/KAM/WBA/multisurface)
  adopt this substrate now, or does it ship purely as an additive capability? Decision
  gates Phase 4's scope.
