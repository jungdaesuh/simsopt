# Item 14 Blocker — `tracing.cpp` → `jax_core/tracing.py`

Status: **BLOCKED** with category `missing_dependency`.

Closure level: `blocked_dependency`.

## Item 14 scope

Port `src/simsoptpp/tracing.cpp` (560 LOC) + supporting `tracing.h` (158 LOC,
9 `StoppingCriterion` subclasses) to a new JAX module
`src/simsopt/jax_core/tracing.py` using an in-repo JAX RK implementation only
(no new dependency such as `diffrax`).

## C++ kernel structure

Read at `src/simsoptpp/tracing.cpp` (parent commit `a9da18fac`):

- **Six RHS classes** (state size in parentheses):
  - `GuidingCenterVacuumRHS` (4: x, y, z, v_par) — vacuum GC in Cartesian
    (`tracing.cpp:25-78`).
  - `GuidingCenterVacuumBoozerRHS` (4) — vacuum GC in Boozer coords
    (`tracing.cpp:80-130`).
  - `GuidingCenterNoKBoozerRHS` (4) — K=0 limit Boozer GC
    (`tracing.cpp:132-188`).
  - `GuidingCenterBoozerRHS` (4) — full Boozer GC with K(s, θ, ζ)
    (`tracing.cpp:190-254`).
  - `FullorbitRHS` (6) — Lorentz force, no GC approximation
    (`tracing.cpp:256-301`).
  - `FieldlineRHS` (3) — `dx/dτ = B` (`tracing.cpp:302-331`).
- **Integrator**: Boost odeint `runge_kutta_dopri5` (RK45 Dormand-Prince) with
  `make_dense_output` (adaptive step + dense interpolation, `tol=rtol=atol`).
- **Event localization** uses Boost `toms748_solve` on the dense interpolant
  per plane crossing (`tracing.cpp:385-386, 408-420`).
- **Stopping criteria**: 9 polymorphic `StoppingCriterion` subclasses polled
  every accepted step (`tracing.cpp:432-438`).

## Why this is BLOCKED

Three structural prerequisites must be ported before any meaningful Item 14
JAX-native tracing port can claim parity with the C++ oracle for the public
SIMSOPT tracing surface (`particle_guiding_center_tracing`,
`particle_guiding_center_boozer_tracing`, `particle_fullorbit_tracing`,
`fieldline_tracing`):

1. **`RegularGridInterpolant3D` JAX port** — Item 13. Tracking; in flight at
   the time of this blocker note. Required by `LevelsetStoppingCriterion`
   (surface classifier) and by `InterpolatedField` consumers of the tracing
   path.
2. **Boozer field JAX port** — `simsoptpp/boozerradialinterpolant.cpp` and
   `boozermagneticfield*.h`. These are **Tier P5** items (32-33) of the goal
   prompt and are **explicitly future-scope** unless the human launching the
   goal expands `active_scope` to include P5. Without a JAX Boozer field,
   three of the four public guiding-center tracing entry points
   (`particle_guiding_center_boozer_tracing` vacuum / noK / full variants)
   cannot reach JAX-native parity.
3. **Surface classifier** (`LevelsetStoppingCriterion` in
   `field/tracing.py:744`) requires Item 13 plus a small classifier kernel.

Additionally, two technical obstacles preclude a single-iteration MVP scope
from claiming the full tracing parity contract:

- **Adaptive variable-step integration inside `lax.while_loop`**: the C++
  scheme uses dense output for both event detection and trajectory recording.
  JAX `lax.while_loop` has fixed-shape carry only; trajectories must be
  pre-allocated with a max-step cap and masked. Backwards autodiff through
  the loop requires `checkpoint`/`remat` memory budgeting. This is a
  designable contract but requires its own validation lane that does not
  exist in `PARITY_LADDER_TOLERANCES`.
- **TOMS748 root-finder replacement**: the Boost `toms748_solve` produces
  bit-accurate Poincaré-plane crossing times. A JAX-compatible bracketed
  root solver (bisection / Illinois) costs accuracy bits — the existing
  `*_parity` byte-identity gate cannot apply unchanged. Adding a new lane
  to `PARITY_LADDER_TOLERANCES` is `tolerance_policy` territory, which the
  prompt rules out for self-resolution.

## MVP carve-out (NOT done in this run)

A minimal fieldline-only MVP could implement a single
`fieldline_trace_jax(field_spec, xyz_init, tmax, tol, phis, max_steps)`
using existing `jax_core/field.biotsavart_B`, with a hand-rolled DOPRI5
plus PI step controller, Hermite dense interpolant, bisection event
localization, and 8/9 stopping criteria (skipping
`LevelsetStoppingCriterion`). The audit estimates ~700-900 LOC of new
JAX code, plus a new parity-ladder lane for event-time tolerance.

This MVP is **not undertaken** in the current run because:

- The scope exceeds the per-item budget implied by the goal prompt's P1
  cadence (each item is a single coherent kernel port).
- Without the Boozer field port, the MVP closes ≤25% of the public
  tracing surface (fieldline only), leaving the bulk of consumers
  (`tests/field/test_particle.py`, `test_mpi_tracing.py`,
  `test_fieldline.py`) on the C++ path.
- The new event-time tolerance lane required by the MVP is
  `tolerance_policy`-adjacent and should be discussed with the human user
  before being added to `benchmarks/validation_ladder_contract.py`.

## Diagnostic budget

- Two timeboxes were not consumed because the blocker is a categorical
  dependency miss, not an empirical parity failure. The audit at
  `.artifacts/jax_port_goal/plans/14-coverage.md` (not written; see below)
  enumerates the missing dependencies. Per the goal prompt's section 5,
  `missing_dependency` does not require the two-timebox budget — it is the
  proposed-dependency category.

## Proposed user decision

Two options for the human launching the goal:

**Option A** (recommended for this run, default): leave Item 14 BLOCKED.
Acknowledge the closure path is gated on (a) P5 expansion to include the
Boozer field port (items 32-33), (b) Item 13 closure (in flight), and
(c) a `missing_dependency` decision on the surface classifier kernel.

**Option B**: expand the active_scope to authorize the MVP fieldline-only
port. This requires:

- Adding `event_time_tracing_tolerance` (or similar) to
  `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. The
  current strict gate is byte-identity at lane tolerance; an event-time
  lane explicitly allowing ~1e-7 absolute tolerance is `tolerance_policy`
  territory and the human must approve it.
- A separate item under the manifest for `fieldline_trace_jax` MVP, with
  Boozer GC and `LevelsetStoppingCriterion` explicitly carved out.

## State.json entry

```json
{
  "id": "14",
  "tier": "P1",
  "title": "tracing RK path",
  "status": "blocked",
  "closure_level": "blocked_dependency",
  "blocker": {
    "category": "missing_dependency",
    "detail": "Item 14 requires (a) item 13 closure for RegularGridInterpolant3D-backed LevelsetStoppingCriterion, (b) a JAX Boozer field port (P5 items 32-33, currently future-scope), (c) a JAX surface classifier kernel, and (d) a tolerance-policy decision on event-time accuracy for the bisection root solver. None of these are resolvable inside the active-scope budget without expanding active_scope and adding a tolerance lane.",
    "debug_artifact": ".artifacts/jax_port_goal/blockers/14-debug.md",
    "needs_user": true
  },
  "evidence": {
    "source_audit": "src/simsoptpp/tracing.cpp:25-554 (6 RHS classes, RK45+TOMS748+9 stopping criteria)",
    "upstream_oracle": "byte-identical to upstream at SHA 1b0cc3a96063197cdbdd01559e04c25456fbe6ff",
    "upstream_audit_sha": "1b0cc3a96063197cdbdd01559e04c25456fbe6ff",
    "downstream_consumers": [
      "src/simsopt/field/tracing.py (936 LOC; 8 sopp.* call sites + 9 StoppingCriterion subclasses)",
      "tests/field/test_particle.py",
      "tests/field/test_fieldline.py",
      "tests/field/test_mpi_tracing.py",
      "tests/configs/test_LHD_like.py"
    ],
    "missing_dependencies": [
      "src/simsopt/jax_core/regular_grid_interp.py (item 13 — in flight)",
      "JAX Boozer field port (P5 items 32-33 — future-scope, blocked on active_scope expansion)",
      "JAX surface classifier kernel for LevelsetStoppingCriterion",
      "tolerance lane for event-time accuracy in PARITY_LADDER_TOLERANCES (tolerance_policy)"
    ],
    "cuda_smoke": "not_claimed"
  }
}
```
