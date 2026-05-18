# X2 — `while_loop` reverse-mode AD fix (H-7)

Scope: triage the 6 `lax.while_loop`-bound integrators flagged in
`.artifacts/jax_convention_review_2026-05-16/00_SYNTHESIS.md` §Theme 2,
fix all Class-C docstring claims, and ship one Class-A `scan` PoC.

## §1 Phase A — Triage

Static iteration count = ceiling is a Python int known at trace time
(loop terminates only by counter or convergence). Dynamic = bound is
also static (`max_steps`) but the loop terminates early on a runtime
predicate (`t < tmax`, `step_count < max_steps`, `stop` flag, etc.),
i.e. the *effective* iteration count is data-dependent.

| # | Function                       | File:line                                   | Iters     | Grad consumer in repo? | Class |
|---|--------------------------------|---------------------------------------------|-----------|------------------------|-------|
| 1 | `_integrate_tangent_map`       | `src/simsopt/jax_core/magnetic_axis_helpers.py:515` | dynamic   | none                   | C     |
| 2 | `trace_fieldline`              | `src/simsopt/jax_core/tracing.py:1154`              | dynamic   | none                   | C     |
| 3 | `trace_guiding_center`         | `src/simsopt/jax_core/tracing.py:1694`              | dynamic   | none                   | C     |
| 4 | `_run_dopri5_4state` (Boozer fast path) | `src/simsopt/jax_core/tracing.py:1841`     | dynamic   | none                   | C     |
| 5 | `trace_guiding_center_boozer` (events path) | `src/simsopt/jax_core/tracing.py:2636` | dynamic   | none                   | C     |
| 6 | `trace_fullorbit`              | `src/simsopt/jax_core/tracing.py:3174`              | dynamic   | none                   | C     |
| 7 | `bracket_root_jax`             | `src/simsopt/jax_core/tracing.py:794` (pre-PoC) / `:790` (post-PoC, now `scan`) | **static** (`max_iters` Python int; cond is just `i < max_iters`) | none                   | **A** |

Grep audit for `jax.grad`, `jax.vjp`, `jax.value_and_grad`,
`jax.jacrev` consumers of any of these 7 names returned zero hits in
`src/`, `tests/`, and `benchmarks/` (third-party `.miniforge` /
`.conda` hits excluded). The H-7 finding "convention review says
`on_axis_iota_rk` advertises gradient support" is a docstring claim,
not a live consumer.

Note on user-vs-current numbering: the user task counted "6
integrators" and described the Boozer entry as `tracing.py:1973/2799`
(both no-events + events `while_loop`s in `trace_guiding_center_boozer`).
The current HEAD splits the no-events fast path into a shared
`_run_dopri5_4state` driver at 1841 and the with-events code at 2636,
so 6 of the 7 enumerated rows above all reduce to the original 6
high-traffic integrators (`bracket_root_jax` is the 7th, the one I
converted).

## §2 Phase B — Docstring fixes applied

For each Class-C integrator I appended a "Notes" block at the end of
the public docstring stating that reverse-mode AD raises and forward-
mode AD is supported. The text matches the actual JAX error class
(`ValueError`, not `TypeError` as the convention review's prose
claimed — I verified by running `jax.grad` on a minimal `while_loop`
mirror and capturing the live error class).

Edits applied (file:line, on post-edit numbering):

- `src/simsopt/jax_core/magnetic_axis_helpers.py:14-23` — module-level
  docstring no longer claims "differentiable through field DOFs as long
  as the user supplies a JAX-traceable field-evaluation callback"; it
  now states the forward-vs-reverse-mode contract explicitly and points
  at JBP-3.3.
- `src/simsopt/jax_core/magnetic_axis_helpers.py:463-472` —
  `_integrate_tangent_map` docstring AD note added.
- `src/simsopt/jax_core/magnetic_axis_helpers.py:578-595` —
  `on_axis_iota_rk` docstring AD note added at end of Notes block.
- `src/simsopt/jax_core/tracing.py:838-854` — `trace_fieldline`
  docstring AD note added.
- `src/simsopt/jax_core/tracing.py:1383-1399` — `trace_guiding_center`
  docstring AD note added.
- `src/simsopt/jax_core/tracing.py:2300-2317` —
  `trace_guiding_center_boozer` docstring AD note added.
- `src/simsopt/jax_core/tracing.py:2855-2872` — `trace_fullorbit`
  docstring AD note added.

`ruff check` and `ruff format` pass on both modified files.

## §3 Phase C — PoC: `bracket_root_jax` `while_loop` → `scan`

### Choice rationale

`bracket_root_jax` is the lowest-risk Class-A candidate:

- the existing `cond` is literally `i < max_iters` — no convergence
  predicate, no dynamic termination, just a counter;
- the body already early-terminates by `lax.cond(converged, no_eval,
  eval)` and applies `where`-noops to the carry once `converged`, so
  the existing semantics already match "execute all `max_iters`
  iterations, but noop after convergence" — exactly the contract
  `scan` provides;
- there is exactly one carry tuple and no per-iteration output of
  interest, so the conversion is `xs=None, length=max_iters` with the
  body returning `(carry, None)`.

### Diff (semantic)

Pre-PoC (HEAD `src/simsopt/jax_core/tracing.py:697-797`): seven-tuple
carry including `i: int32`, `cond(carry) = i < max_iters`, body
increments `i` and returns the seven-tuple. Final unpack discards `i`.

Post-PoC: six-tuple carry (drops `i` — `scan` tracks iteration count
internally), body signature changed to `body(carry, _x) ->
(carry, None)`, final call is

```python
(_a, _b, _fa, _fb, t_best, f_best), _ = jax.lax.scan(
    body, init, xs=None, length=int(max_iters)
)
```

No other changes — the false-position update, the `converged` mask,
the `keep_left` Illinois update, and the `improves_best` book-keeping
are bit-identical. The docstring Notes block now states reverse-mode
AD is **supported** under `scan` instead of forbidden.

### Verification

1. **Existing tests** (no behavior change expected): all
   `bracket_root_jax`-related tests pass.

   ```
   .conda/jax/bin/python -m pytest \
     tests/jax_core/test_tracing_jax_item14.py \
     tests/jax_core/test_tracing_jax_phi_events.py \
     tests/jax_core/test_tracing_jax_levelset_events.py -x -v
   ```

   23 passed, 0 failed. The 3 direct `bracket_root` tests
   (`finds_zero_crossing_within_tolerance`,
   `uses_false_position_candidate_for_linear_residual`,
   `returns_false_when_no_sign_change`) all pass at unchanged
   tolerance gates. The 10 `phi_events` tests and 7 `levelset_events`
   tests — all of which route through `bracket_root_jax` as the inner
   event localizer of `trace_fieldline` — also pass.

   The `on_axis_iota_rk` test set
   (`tests/field/test_magnetic_axis_helpers_jax_item21.py`) also still
   passes 15/15 (regression check for the docstring-only edits).

2. **Reverse-mode AD smoke test** (throwaway, not committed):
   constructed `make_objective(c) = bracket_root_jax(t |-> t - c,
   0, 1, ...)`, called `jax.grad(make_objective)(0.7)`. Result: `1.0`
   (analytic dt/dc), matches `jax.jvp` (`1.0`) and central finite
   difference (`1.0000000000287557`).

   For comparison, on a minimal `while_loop` mirror of the same body
   shape (counter + accumulator with the same data flow), the same
   `jax.grad` call raises
   `ValueError: Reverse-mode differentiation does not work for
   lax.while_loop or lax.fori_loop with dynamic start/stop values. Try
   using lax.scan, or using fori_loop with static start/stop.`
   So the fix is observable and the contract change is real.

3. **No value drift**: the existing pytest suite asserts a specific
   `t_star` value to machine precision (e.g.
   `np.testing.assert_allclose(float(t_star), 0.7, rtol=0.0,
   atol=1e-15)`); these gates pass, so the `scan` version produces
   bit-identical output on the converged path.

4. **Ruff**: `ruff check` and `ruff format` both pass on
   `tracing.py`.

## §4 Phase D — Deferred follow-up

The remaining 6 `while_loop` integrators are all Class-C with the
docstring contract now correct. They can stay on `while_loop` until a
grad consumer materializes. Risk and effort for upgrading them to
`scan` or `custom_vjp`-IFT:

| Function | Effort | Risk | Notes |
|----------|--------|------|-------|
| `_integrate_tangent_map` (on_axis_iota_rk) | ~0.5 d | low | Carry is 5 fields. Body is one DOPRI5 step + PI controller. `max_steps=10000` budget — `scan` over 10k iterations with 5-field carry is `~80kB` per RHS state — acceptable. Risk: must verify `accepted` semantics match (rejected step `where`-noop). Has a closed-form FD oracle (`compute_on_axis_iota`) so value diff testable cheaply. |
| `trace_fieldline` | 1-2 d | medium | 14-field carry (incl. trajectory `(max_steps+1, 4)` and `(max_phi_hits, 5)` buffers). `scan` over 10k+ steps will allocate up to a few MB. Inside the body sits the (now `scan`-based) `bracket_root_jax` — that nesting will work cleanly. Risk: `phis_arr` `for i in range(num_phis)` loop is Python-unrolled inside the body; this stays unchanged but the unroll factor compounds with the outer `scan` length. Watch compile time. |
| `trace_guiding_center` | 1-2 d | medium | Same architecture as `trace_fieldline` but 4-state RHS. Add roughly +1 carry field. |
| `_run_dopri5_4state` (Boozer fast path) | 0.5 d | medium-low | Smaller carry (10 fields, no phi-hits buffer). Used by `trace_guiding_center_boozer(zetas=None, stopping_criteria=())`. |
| `trace_guiding_center_boozer` (events path) | 1-2 d | medium | Same scope as `trace_fieldline` with Boozer state. |
| `trace_fullorbit` | 1-2 d | medium | 6-state RHS, 8-wide `phi_hits` rows. Same scope as `trace_fieldline`. |

Total deferred effort: 5-8 person-days. Trigger to schedule: any
downstream consumer wires `jax.grad`/`jax.vjp` through one of these.

Recommended order if scheduled:

1. `_integrate_tangent_map` — smallest carry, has a closed-form CPU
   oracle, fastest validation;
2. `_run_dopri5_4state` — smaller-carry variant of the trace drivers,
   exercises the trajectory-buffer `scan` pattern;
3. `trace_fieldline` — broadest test surface, validates phi-event
   localizer composition with the outer `scan`;
4. `trace_guiding_center`, `trace_guiding_center_boozer`,
   `trace_fullorbit` — copy of the `trace_fieldline` pattern with
   state-shape tweaks.

For all 5 remaining drivers, the recommended technique is **`lax.scan`
with `mask` + `where`-noop** (Class-A), not `custom_vjp`-IFT. The
adaptive integrator state is path-dependent — there is no fixed-point
formulation to exploit, so IFT is the wrong tool. The body is already
written in a "noop-once-stopped" style via the existing `stop` flag,
so the `scan` rewrite reduces to (a) dropping the cond, (b) adding an
external mask in the body that gates carry updates after `stop`
becomes true, and (c) `length=max_steps`. Carry size and `max_steps`
are both static, so compile cost is bounded; **runtime cost will go up
linearly with `max_steps`** because the rejection / `stop` mask cannot
skip iterations the way `while_loop` does, so a 10k-budget trace that
currently terminates in 200 steps will materially slow down on hot
paths.

That runtime regression is the real cost calculus, not the engineering
effort. Until a real reverse-mode AD consumer exists, the current
docstring contract is the correct disposition.

## §5 Verification summary

- Phase A: 7 functions enumerated and classified above. Grep audit
  confirms zero in-repo `jax.grad`/`jax.vjp`/`jax.value_and_grad`/
  `jax.jacrev` consumers on any of them.
- Phase B: 7 docstring edits (1 module + 6 functions). `ruff check` +
  `ruff format` clean on both modified files. Tests pass.
- Phase C: `bracket_root_jax` converted from `while_loop` to `scan`.
  - 23 existing tests pass
    (`tests/jax_core/test_tracing_jax_item14.py`,
    `tests/jax_core/test_tracing_jax_phi_events.py`,
    `tests/jax_core/test_tracing_jax_levelset_events.py`).
  - 15 existing tests pass for `on_axis_iota_rk`
    (`tests/field/test_magnetic_axis_helpers_jax_item21.py`).
  - Throwaway grad smoke test verified `jax.grad(bracket_root_jax)`
    succeeds (returns analytic dt/dc = 1.0; matches jvp and FD).
  - Throwaway proof verified that the equivalent `while_loop`
    structure still raises `ValueError`, confirming the contract
    change is observable.
- Phase D: 6 deferred Class-C → Class-A migration targets documented
  with effort/risk; no in-repo grad consumer forces the work.

## Files touched

- `src/simsopt/jax_core/magnetic_axis_helpers.py` — docstring AD
  notes (module-level + `_integrate_tangent_map` +
  `on_axis_iota_rk`).
- `src/simsopt/jax_core/tracing.py` — docstring AD notes (4 trace
  functions) + `bracket_root_jax` `while_loop`→`scan` PoC.

## Files **not** touched

The 6 deferred Class-C `while_loop` call sites in the listed
integrators remain on `while_loop` by design. The convention-review
flag on the `on_axis_iota_rk` module docstring is closed: the false
"differentiable through field DOFs" claim is gone and the actual
forward-vs-reverse-mode contract is now written down.
