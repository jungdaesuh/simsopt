# Pole-2 Compile-Breadth Kernelization Plan (single-stage GPU value/grad)

> Created 2026-06-23. Status: **DESIGN-NOTE DRAFT (not implementation signoff).** Grounded
> against the working tree at HEAD `29fd5d146` (dirty, concurrent codex edits — anchor on
> symbol names, not line numbers). Focused companion to
> `docs/torax_style_host_controlled_lbfgs_kernelization_implementation_plan.md` (outer-loop
> host control) and `docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md`
> (adjoint, RETIRED → keep dense+chunk).
> Doc-review pass 2026-06-23: every repo-local file:line re-verified against the live tree
> (several had drifted — fixed); `newton_exact_traceable` (the sanctioned traceable
> exact-Newton) added; the strict-jax framing corrected from an open question to a settled
> by-design ban; official JAX docs checked for `lax.while_loop`, `jit(static_argnames)`,
> `custom_vjp`/`custom_root`, persistent cache, compile logging, and XLA flags.

## Purpose

The production-scale (mpol10) single-stage GPU benchmark is blocked by **pole 2**: the
per-evaluation value+gradient program takes ~tens-of-minutes-to-an-hour to *construct +
XLA-compile* with the GPU idle, before a single optimizer step runs. This plan localizes
and (if tractable) reduces that compile cost. It exists because pole 2 — unlike pole 1
(the dense-adjoint constant-fold, already fixed) — is **not** a single hot spot; it is the
**breadth** of the fused per-eval graph, and a 2026-06-23 code trace ruled out the
"obvious" lever (an unrolled inner Newton). So the next step must be *measurement-first*,
not a blind rewrite.

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
  bans host-driving the inner Newton and supplies a traceable rolled path instead — which means
  pole-2 narrowing must stay **within a traced graph** (breadth reduction) unless the strict-jax
  no-host-inner-loop policy is explicitly relaxed. [verified]
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
  (`boozer_surface.py:199` `EXACT_FACTORIZATION_BACKEND = "operator-gmres"`); dense gate is
  caller-blind (`optimizer.py:4510`); dense chunk `batch_size=8`
  (`optimizer.py:3608`, `lax.map`). [verified, Gate-4 + this pass]
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

## Rationale

Pole 1 was a single oversized sub-graph (the dense adjoint) → one env-var fix. Pole 2 is
diffuse **breadth**, and the two highest-leverage breadth reducers are already in place
(outer loop host-driven; adjoint implicit). The 2026-06-23 trace removes the remaining
"easy" hypothesis (unrolled Newton), so the honest path is: **measure where the breadth and
compile time actually concentrate at mpol10**, then narrow the dominant piece. The most
torax-faithful narrowing (host-driving the inner Boozer loop) is banned by strict-jax by
design (`newton_exact` fallback guard; `newton_exact_traceable` is the sanctioned traceable
path), so the plan must reduce traced breadth *within* a single graph — or escalate an
explicit strict-jax policy change. Measuring first avoids a large rewrite that may not move
the dominant term, and may instead prove the floor is irreducible (a legitimate,
decision-useful outcome).

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
         the traceable LM/adam runner reaches the rolled backtracking (`optimizer.py:4032/4120`)
         or the Python `for range` one (`:4055`); audit `:3615` and `:4226` for traceability.
         Record the verdict (expected: rolled path; flag if not).
2. **Decide the narrowing strategy from the measurement**
   - [ ] If one sub-computation dominates (e.g. BiotSavart field eval), plan to extract it as a
         separate static-shape **cached kernel** called from the host-orchestrated step
         (optax "jit the step, not the macro"), reusing the static-arg / persistent-cache
         machinery from the torax_style plan.
   - [ ] Honor the policy boundary: strict-jax bans host-driving the inner Newton
         (`newton_exact:5124` guard) and provides `newton_exact_traceable:5402` as the
         traceable path — so default narrowing stays **within a traced graph**. Decide whether
         within-graph breadth reduction suffices, or whether a strict-jax policy change (to
         allow host-controlled inner stepping) is warranted (larger decision; Open Questions).
   - [ ] Write the design twice (per software-design Tier-2 gate): (a) within-graph breadth
         split vs (b) strict-jax policy change enabling host-controlled inner stepping; pick
         with the measurement (default toward (a)).
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
  path). Mitigation: target within-graph breadth reduction first; only if it is insufficient,
  escalate a strict-jax policy change as an explicit, recorded decision (not silently bypassed).
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
- [ ] A recorded decision: narrow-within-graph vs strict-jax policy change, **or**
      "compile floor irreducible → use mpol≤N / native" with evidence.
- [ ] If narrowing implemented: mpol10 compile bounded + parity preserved + Crucible PASS.

## Open Questions

- Under current strict-jax policy, within-graph breadth reduction is the only narrowing allowed
  — host-driving the inner Newton is banned by design (`newton_exact:5124` guard;
  `newton_exact_traceable:5402` is the sanctioned traceable path). If within-graph reduction
  proves insufficient, is relaxing the strict-jax no-host-inner-loop policy on the table?
  (Owner: user / jax-policy.)
- Should this plan be **merged into**
  `torax_style_host_controlled_lbfgs_kernelization_implementation_plan.md` rather than stand
  alone? It adds the pole-2-at-mpol10 grounding (decomposed lane also bound; inner Newton
  already rolled; strict-jax ban + traceable path) but overlaps that plan's mechanics. (Owner: user.)
- Is the per-eval breadth dominated by BiotSavart eval, residual/Jacobian construction, or the
  adjoint? (Resolve in Phase 1 — determines the entire narrowing design.)
