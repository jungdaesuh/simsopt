# Pole-2 Compile-Breadth Kernelization Plan (single-stage GPU value/grad)

> Created 2026-06-23. Status: **DESIGN-NOTE DRAFT (not implementation signoff).** Grounded
> against the working tree at HEAD `dd978a21e` (dirty, concurrent codex edits — anchor on
> symbol names, not line numbers). Focused companion to
> `docs/torax_style_host_controlled_lbfgs_kernelization_implementation_plan.md` (outer-loop
> host control) and `docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md`
> (adjoint, RETIRED → keep dense+chunk).
> Doc-review pass 2026-06-23: changed repo-local anchors re-verified against the live tree;
> `newton_exact_traceable` (the sanctioned traceable exact-Newton) added; the strict-jax
> framing corrected from an open question to a settled by-design ban; official JAX docs
> checked for `lax.while_loop`, `jit(static_argnames)`, `custom_vjp`/`custom_root`,
> persistent cache, compile logging, and XLA flags.

## Purpose

The production-scale (mpol10) single-stage GPU benchmark is blocked by **pole 2**: the
per-evaluation value+gradient program takes ~tens-of-minutes-to-an-hour to *construct +
XLA-compile* with the GPU idle, before a single optimizer step runs. This plan localizes
and (if tractable) reduces that compile cost. It exists because pole 2 — unlike the earlier
dense-adjoint constant-fold failure, which is suppressible per run but still needs a durable
default policy fix — is **not** a single hot spot; it is the **breadth** of the fused
per-eval graph, and a 2026-06-23 code trace ruled out the "obvious" lever (an unrolled inner
Newton). So the next step must be *measurement-first*, not a blind rewrite.

## Goals

- A **measured localization** of pole-2 compile cost at mpol10: which compiled unit (K1
  forward solve vs K2 value+adjoint) dominates, which sub-computation inside it dominates
  (BiotSavart field eval vs residual/Jacobian vs adjoint), and how compile time/HLO size
  scales mpol6→mpol8→mpol10.
- Either (a) a **narrowing** that brings the mpol10 per-eval compile wall down to minutes
  and roughly flat vs mpol, behind a default-off flag, with machine-precision gradient
  parity preserved; or (b) a defensible **"compile floor is irreducible"** finding with
  evidence and a production recommendation (mpol≤N on GPU, or CPU/native at full mpol).
- No regression to the already-working pieces: host-driven outer L-BFGS, operator-GMRES +
  `custom_vjp` adjoint, the `batch_size=8` dense chunk.

## Non-Goals

- Re-deriving outer-loop host control / static-arg + persistent-cache machinery — owned by
  `torax_style_host_controlled_lbfgs_kernelization_implementation_plan.md`.
- Flipping the inner-Boozer adjoint to matrix-free — Gate-4 is RETIRED; keep dense+chunk
  (`docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md`).
- Making the full single-stage outer loop one on-device graph (already rejected; caused the
  422 GiB ondevice compile OOM — memory `project_ondevice_compile_blowup_root_cause`).
- "Roll the inner Newton" — **already rolled** (see Current Context); not an available lever.
- nphi reduction as a sole fix — measured nphi64 and nphi127 runs both remained
  construction/compile-stalled, so grid reduction has not cleared pole 2. Whether the
  remaining breadth is dominated by mode-count (`mpol`/`ntor`) residual/Jacobian structure,
  BiotSavart geometry, or adjoint work is a Phase-1 measurement question, not a pre-decided
  conclusion.

## Current Context (verified in this repo 2026-06-23 unless marked)

- **Pole 2 hits BOTH GPU lanes at mpol10.** The decomposed lane (`scipy-jax-decomposed`)
  was compile-stalled on every mpol10 run this session; the fused lane (`scipy-jax`) showed
  the identical GPU-0% / log-empty / ~7-min construction signature before it was killed.
  So decomposing the outer loop did **not** clear pole 2. [measured, this session; pod now stopped]
- **The traceable inner Boozer solve is ALREADY rolled (`lax.while_loop`), not Python-unrolled.**
  `_make_traceable_levenberg_marquardt_runner` (`optimizer.py:1886`, its `lax.while_loop` at
  `:2017`) and the traceable adam runner (`lax.while_loop` `:1872`, jitted `:1875`) are the
  jitted solvers; there is also a trace-safe exact-Newton `newton_exact_traceable` (`:5402`).
  Therefore compile size should not grow by Python-unrolling the Newton iteration count —
  pole 2 is graph **breadth**, not loop depth. [verified against code; JAX docs confirm
  `lax.while_loop` lowers as a single loop op rather than unrolling Python loop bodies]
- **Host-driving the inner loop is banned by strict-jax BY DESIGN; the sanctioned path is
  traceable+rolled.** `newton_exact` (`optimizer.py:5124`) is a host
  `while nit < maxiter and float(norm) > tol` loop (host sync via `float(norm)`) — exactly the
  torax pattern — but it begins with `raise_if_strict_jax_fallback(... "host-controlled exact
  Newton loop")` (`:5139`); it exists only for dense compat metadata. Its sanctioned counterpart
  `newton_exact_traceable` (`:5402`) has **no** strict guard, is rolled, and is matrix-free
  (JVP, no dense Jacobian) via `_make_traceable_exact_newton_runner`. So strict-jax deliberately
  bans host-driving the inner Newton and supplies a traceable rolled path instead. This plan keeps
  pole-2 narrowing inside that current policy. [verified]
- **Two backtracking variants coexist:** rolled `lax.while_loop` (`optimizer.py:4032,4120`)
  and a Python `for iteration in range(_NEWTON_BACKTRACKING_MAX_STEPS)` (`optimizer.py:4055`,
  `_NEWTON_BACKTRACKING_MAX_STEPS = 8` at `:376`, unrolled). Which one the traceable path
  traces is **unconfirmed** (Phase-1 item). [verified existence]
- **Per-eval kernel composition (the breadth):** fused `_value_and_grad_for`
  (`surface_objectives_traceable.py:1390`, jit at `:1428`) wraps K1 = `_forward_result_for`
  (`:1336`) and K2 = `_total_gradient_for` (`:1373`), joined by a device `lax.cond` (`:1414`).
  The decomposed lane splits K1 / K2 = `_solved_state_value_and_grad_for` (`:1456`), built by
  `_build_decomposed_coil_host_value_and_grad` (`single_stage_banana_example.py:9697`) and
  host-glued by `if host_bool(...)` (`:9716`). So the OUTER K1∪K2 fusion is already splittable;
  remaining breadth is *inside* K1 (forward solve + BiotSavart field eval + residual/Jacobian)
  and K2 (objective + adjoint). [verified, prior crucible + this pass]
- **Already-applied levers:** adjoint is operator-GMRES + `custom_vjp`/IFT
  (`boozer_surface.py:199` `EXACT_FACTORIZATION_BACKEND = "operator-gmres"`); the dense
  materialization gate is byte/dimension-aware but not compile-breadth-aware
  (`optimizer.py:4510-4516`); dense residual-Jacobian assembly is chunked with
  `batch_size=8` (`optimizer.py:3608`, `lax.map`). [verified, Gate-4 + this pass]
- **torax reference levers** (for the template, local upstream checkout
  `/Users/suhjungdae/code/opensource/torax`): host outer loop calling jitted step
  (`torax/_src/orchestration/run_loop.py:113`); rolled Newton `lax.while_loop`
  (`torax/_src/solver/jax_root_finding.py:108`); implicit-diff adjoint
  `lax.custom_root` (`torax/_src/solver/jax_root_finding.py:129`, `tangent_solve`
  at `:133`); bounded solve kernels with `static_argnames`
  (`torax/_src/fvm/optimizer_solve_block.py:43`); compile-count test
  `get_number_of_compiles` (`torax/_src/jax_utils.py:147`). torax state is ~25–100
  cells — intrinsically smaller than our BiotSavart-over-thousands-of-quadrature-points
  per-eval. [verified, this session]
- **Official JAX-doc constraints for this plan:** `lax.while_loop` is the correct primitive
  for rolled iterative kernels because native Python loops in a jitted function are unrolled
  into large computations, while `lax.while_loop` lowers as one loop op and requires fixed
  carry shape/dtype. It is forward-mode (JVP) differentiable but **not** reverse-mode
  (VJP/`grad`) differentiable — exactly why the inner-solve adjoint uses `custom_vjp`/operator-GMRES
  (implicit differentiation) instead of back-propagating through the loop. `static_argnames` makes arguments compile-time constants and therefore
  changing them intentionally recompiles. Persistent compilation cache keys include the HLO,
  jaxlib version, relevant XLA flags, device topology, and other execution facts, so Phase-1
  measurements must label cold-cache vs warm-cache runs and keep XLA flags fixed. `XLA_FLAGS`
  should be exported before any JAX backend initialization.

## 4-agent breadth analysis + ordered cure (2026-06-23)

A 4-subagent READ-ONLY analysis (breadth localization; staging-under-strict-jax; torax/optax
template; cache + measurement) converged on a concrete diagnosis. Headlines:

- **Crux RESOLVED — the cure is WITHIN strict-jax; no policy change.** strict-jax only bans
  host-stepping the *inner* Newton (O(40) device↔host syncs/outer-step; the `newton_exact:5124`
  guard). The cure moves a *single* scalar — the Boozer success gate — to the host between two
  separately-compiled kernels (K1 forward, K2 adjoint); the inner Newton stays an on-device
  `lax.while_loop`. This is exactly what the `scipy-jax-decomposed` lane already does
  (`_build_decomposed_coil_host_value_and_grad:9697`, `host_bool:9716`).
- **Core over-fusion (A+B+C agree):** `jax.jit(_value_and_grad_for)`
  (`surface_objectives_traceable.py:1428`) re-fuses K1 (`jitted_forward_result_for`) + the
  success `lax.cond` (`:1414`) + K2 (`compiled_total_gradient_for`) into ONE XLA program; the
  `lax.cond` forces BOTH adjoint branches into the graph.
- **Why decomposing alone did NOT clear pole 2:** the split removes only the OUTER `lax.cond`.
  The residual breadth is a STACK (ranked by Agent A; #1 now confirmed to FIRE under the
  production default — see the Reliability note):

  | # | breadth driver | location | fix |
  |---|---|---|---|
  | 1 | dense Hessian `vmap` (≈1323 HVPs) — **FIRES under production default** (14 MB ≪ 256 MiB GPU cap; no override in `src/`) = pole 1 active in prod | `optimizer.py:4503` (`_apply_column_batched_operator`, `vmap` :4507), existing byte/dimension gate `:4510-4516`, caps `src/simsopt_jax/backend/runtime.py:233-234` | compile-breadth-aware dense-adjoint policy/cap → operator-GMRES (durable pole-1 fix — REQUIRED; manual `byte=0` was per-run only) |
  | 2 | K1 carries a SECOND operator-solve subgraph (warm-start predictor `_traceable_predict_warmstart_x`) on top of the forward solve | `_traceable_result_linear_solve_factors`→None is DELIBERATE — runtime adjoint stays operator-backed so the compiled value/grad stages NO dense LU (staging it would itself be pole-1 breadth), `surface_objectives_traceable.py:604-612` | CONFIRMED: a distinct 2nd operator solve (`_traceable_solve_linearization:1088`, RHS `-forcing` = residual-vs-coils JVP ≠ adjoint RHS → NOT dedupe-able). Remedy: host-side warm-start, or shrink each solve via #1/#3; reuse-factors N/A |
  | 3 | BiotSavart point-axis `vmap` remains a Phase-1 compile suspect; point chunking already exists and must be verified/tuned for the production lane | `_point_chunk_reduce` (`src/simsopt_jax/core/biotsavart.py:316-351`), point `vmap` (`:486-494`), defaults/env (`src/simsopt_jax/backend/runtime.py:418-483,492`) | measure effective `point_cs`; tune/promote it only if the active production config is off, disabled by dense-audit mode, or too coarse |
  | 4 | inner `lax.cond` (baseline-vs-general) doubling K1 | `_traceable_forward_result:794`, cond `:853` | host gate / prune |
  | 5 | outer `lax.cond` (success) doubling the adjoint | `_value_and_grad_for:1414` | the decomposed split (host `if`) |

  Reliability (verified 2026-06-23, code-review-fix-loop): weight the ranking on **Agent A**.
  **Agent C's "5659 `jacfwd` in the LM body" is REFUTED** — the LM residual Jacobian is chunked
  `jax.lax.map(jvp_column, eye, batch_size=8)` (`boozer_surface.py:5769`, "instead of jacfwd's
  parallel vmap over all N"); the `:5403` jacfwd is the small *constraint* Jacobian, not the
  residual. **Byte-gate conditional RESOLVED:** production uses the default 256 MiB GPU cap
  (`src/simsopt_jax/backend/runtime.py:234`; no override in `src/`/`examples`/`benchmarks`)
  and the ≈1323²×8≈14 MB adjoint Hessian passes the existing byte/dimension gate →
  **#1 FIRES in production** (pole 1 active by default; the manual `byte=0` only suppressed it
  in the Option-B runs, where the residual stack #2/#3/#4 still poled). The durable fix is not
  "add dimension awareness" — that already exists — but make the dense-adjoint policy
  compile-breadth-aware or lower the production cap for this path.

- **Ordered cure** (cheapest/safest → deepest; each gated on the Phase-1 measurement):
  0. MEASURE (pod-gated; `benchmarks/compile_breadth_probe` is a NET-NEW harness to be created — Agent D produced the spec — run at mpol6/8/10) — confirm the dominant driver.
  1. Durable compile cache (quick win): `benchmarks/run_parity_matrix_pod.sh:45,51` `/tmp`→`/workspace`
     (pay ~1 hr once/pod). Requires K1 callback-free (the `jax.debug.callback` poisoner; tracked by
     `benchmarks/check_cached_kernel_callback_compatibility.py`). Mitigation, not a cure.
  2. Reduce K1's second (warm-start) operator solve (driver #2) — options: host-side warm-start
     prediction, or rely on #1/#3 to shrink each operator solve. ("Reuse forward factors" is N/A — the runtime path is operator-backed by design and stages no dense LU.) Remedy + payoff gated on the measurement.
  3. Verify/tune BiotSavart point chunking (driver #3) — prove the production lane's effective
     `point_cs` and tune/promote it if the active setting is still compiling too broadly.
  4. Compile-breadth-aware dense-adjoint gate/cap (driver #1; durable pole-1 fix).
  5. Promote + harden the decomposed lane / delete the outer-jit re-fusion (drivers #4/#5) +
     dynamic-state explicitization for cross-instance persistent-cache reuse.

## Rationale

Pole 1 was a single oversized sub-graph (the dense adjoint) with a proven env-var suppression,
but the default policy still permits that path. Pole 2 is diffuse **breadth**, and the two
highest-leverage breadth reducers are already in place (outer loop host-driven; adjoint
implicit). The 2026-06-23 trace removes the remaining "easy" hypothesis (unrolled Newton), so
the honest path is: **measure where the breadth and compile time actually concentrate at
mpol10**, then narrow the dominant piece. Host-driving the inner Boozer loop is banned by
strict-jax by design (`newton_exact` fallback guard; `newton_exact_traceable` is the sanctioned
traceable path), and the 4-agent analysis resolved that no policy change is required for the
current cure. This plan therefore reduces traced breadth under the current strict-jax policy.
Measuring first avoids a large rewrite that may not move the dominant term, and may instead
prove the floor is irreducible (a legitimate, decision-useful outcome).

## Assumptions (explicit)

- ASSUMPTION: pole-2 compile cost is dominated by per-eval graph **breadth** (BiotSavart over
  nphi×ntheta×coils + high-mode residual/Jacobian + adjoint), not loop depth. Supported by:
  rolled loops (verified), nphi-reduction not helping (measured), memory
  `project_ondevice_compile_blowup_root_cause` ("breadth"). **To confirm by HLO measurement.**
- ASSUMPTION: K1 and/or K2 individually exceed the practical mpol10 compile budget (else the
  decomposed split would already have cleared pole 2). **To confirm by compiling each alone.**
- ASSUMPTION: gradient parity is preserved by any breadth-narrowing that does not change the
  math (same residual/objective/adjoint, only the jit/host boundary). To gate by parity test.

## Implementation Plan

1. **Measure & localize the breadth (no code changes yet)**
   - [ ] Compile K1 (`_forward_result_for`) ALONE at mpol10 and time construction + XLA
         compile (`JAX_LOG_COMPILES=1`, wall around first call); repeat for K2
         (`_solved_state_value_and_grad_for`). Record which dominates. Run with a fresh
         dump/cache directory for cold-cache timing; repeat with the same directory only when
         intentionally measuring persistent-cache reuse.
   - [ ] Dump HLO (`XLA_FLAGS=--xla_dump_to=…`, exported before backend init) for the per-eval
         kernel at mpol6/mpol8/mpol10; record HLO op count + compile wall vs mpol →
         confirm/deny super-linear breadth scaling and identify the largest sub-computations
         (BiotSavart eval vs residual Jacobian vs adjoint).
   - [ ] Confirm no unrolled Python loop is traced into the per-eval graph: determine whether
         the traceable LM/exact-Newton/adam runners reach the rolled backtracking
         (`optimizer.py:4032/4120`) or the host-only Python `for range` path (`:4055`).
         Record the verdict (expected: rolled path; flag if not).
2. **Decide the narrowing strategy from the measurement**
   - [ ] If one sub-computation dominates (e.g. BiotSavart field eval), plan to extract it as a
         separate static-shape **cached kernel** called from the host-orchestrated step
         (optax "jit the step, not the macro"), reusing the static-arg / persistent-cache
         machinery from the torax_style plan.
   - [ ] Honor the policy boundary: strict-jax bans host-driving the inner Newton
         (`newton_exact:5124` guard) and provides `newton_exact_traceable:5402` as the
         traceable path. The approved narrowing must keep the inner solve traceable/rolled and
         use only the existing host boundary around K1/K2 success.
   - [ ] Write the design twice (per software-design Tier-2 gate) within the current policy:
         (a) dense-adjoint gate/cap + existing decomposed K1/K2 route vs (b) BiotSavart /
         warm-start-solve kernel extraction and tuning. Pick with the measurement.
3. **Implement the chosen narrowing behind a default-off flag** (depends on Phase 2)
   - [ ] Add the narrowed path as an opt-in (default = current behavior, byte-identical when off).
   - [ ] Keep operator-GMRES + `custom_vjp` adjoint and the `batch_size=8` chunk intact.
4. **Re-run the mpol10 benchmark**
   - [ ] decomposed + narrowed lane at mpol10 on april285/299: compile wall, peak GPU,
         optimizer walltime, gradient parity vs CPU/C++. Compare to the pole-2 baseline.

## Validation Plan

- [ ] **Compile-count gate** (torax `get_number_of_compiles` analog,
      `/Users/suhjungdae/code/opensource/torax/torax/_src/jax_utils.py:147`):
      same-shape dynamic inputs → no recompile; optimizer step count does not grow compile
      count or steady memory. Static argument or shape changes may recompile by JAX design and
      should be counted separately rather than treated as regressions.
- [ ] **Compile wall bounded:** mpol10 per-eval construct+compile in minutes (not ~1 hr) and
      roughly flat mpol8→mpol10 (the explicit pole-2 success metric).
- [ ] **Gradient parity:** machine-precision (rel ≤ parity-ladder rtol) vs CPU/C++ on
      warm-start seeds; default-off flag is byte-identical to current.
- [ ] **Memory:** peak GPU + host RSS stable, no regression.
- [ ] **Tests green** under `./.conda-env/bin/python`; **Crucible PASS** on the diff.
- Commands: pod GPU lane via the proven pod-only `/workspace/ip_lane.sh` wrapper documented
  in `RUNBOOK.md` §4.1, or the parity-matrix launcher at mpol10, with `JAX_LOG_COMPILES=1`
  + `XLA_FLAGS=--xla_dump_to=...` set before JAX starts; local adjoint + compile-count tests.

## Risks and Mitigations

- Risk: the most torax-faithful fix (host-driving the inner Newton) is banned by strict-jax by
  design (`newton_exact:5124` guard; `newton_exact_traceable:5402` is the sanctioned traceable
  path). Mitigation: target current-policy breadth reduction; if future evidence reopens
  host-inner stepping, treat it as a new user-visible policy decision rather than part of this
  plan.
- Risk: the breadth is intrinsic (BiotSavart at high resolution) and largely irreducible.
  Mitigation: then the deliverable is a measured **mpol10 GPU compile floor** + recommend
  mpol≤N on GPU or CPU/native at full mpol — an honest, decision-useful finding, not a failure.
- Risk: concurrent codex edits to `optimizer.py` / `surface_objectives_traceable.py`.
  Mitigation: anchor on symbol names; re-read before editing; scope edits.
- Risk: narrowing perturbs gradient parity. Mitigation: parity gate blocks the flip;
  default-off flag.

## Completion Criteria

- [ ] Phase-1 measurement localizes pole-2 compile cost (dominant kernel, dominant
      sub-computation, scaling vs mpol) — recorded with HLO/compile-wall numbers.
- [ ] A recorded decision: which current-policy narrowing path to implement, **or**
      "compile floor irreducible → use mpol≤N / native" with evidence.
- [ ] If narrowing implemented: mpol10 compile bounded + parity preserved + Crucible PASS.

## Open Questions

- ~~Under current strict-jax policy, is relaxing the no-host-inner-loop policy on the table?~~
  **RESOLVED (2026-06-23 4-agent analysis):** the cure is within strict-jax with NO policy
  change. The sanctioned K1/K2 host-dispatch split moves only the single success-gate scalar to
  the host (the inner Newton stays an on-device `lax.while_loop`); breadth is then reduced by the
  ordered stack above (second-solve reduction, point-chunk tuning, compile-breadth-aware
  dense-adjoint gate/cap, decomposed split). Relaxing the no-host-inner-loop policy is NOT required.
- Should this plan be **merged into**
  `torax_style_host_controlled_lbfgs_kernelization_implementation_plan.md` rather than stand
  alone? It adds the pole-2-at-mpol10 grounding (decomposed lane also bound; inner Newton
  already rolled; strict-jax ban + traceable path) but overlaps that plan's mechanics. (Owner: user.)
- Is the per-eval breadth dominated by BiotSavart eval, residual/Jacobian construction, or the
  adjoint? (Resolve in Phase 1 — determines the entire narrowing design.)
