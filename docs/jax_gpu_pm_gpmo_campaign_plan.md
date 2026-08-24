# Permanent-magnet GPMO native-vs-GPU speed: certifying campaign

**Status: DRAFT — NOT FROZEN.** Nothing here is preregistered until this file is committed
unmodified and its sha256 recorded in the receipt. No campaign leg may run before that.

**Drafted:** 2026-08-23, before any certifying evidence exists. **Deliverable of:** task P3.6,
`docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md, Phase 3 task P3.6`. **Protocol authority:**
that plan's §Campaign protocol (same file, lines 186-222), cited clause-by-clause below and never
restated as if it originated here. Charter shape follows
`docs/jax_gpu_finitebuild_native_speed_successor_plan.md`. **Campaign slug:** `pm_gpmo_native_gpu`.

## Scope — what this campaign may and may not mint

Two rungs may mint a certified speed claim; one named member may not.

| Rung | Case | Configuration | Claim sought |
| --- | --- | --- | --- |
| A | `pm-simple-16` | GPMO baseline, nφ=nθ=16, `downsample=4`, K=500; `A_obj` 256 × 43,008, 14,336 dipoles | warm GPU speed at the mirror's own `native_default` scale |
| B | `muse-64` | `ArbVec_backtracking`, nφ=nθ=64, `downsample=10`, K=10000, `max_nMagnets=5000`, `nBacktracking=200`, `nAdjacent=1`; `A_obj` 4096 × 22,590, 7,530 dipoles | warm GPU speed **at the nφ=64 configuration, named** |
| — | `pm4stell-64` | `ArbVec_backtracking`, nφ=nθ=64, `downsample=10`, K=2000, `max_nMagnets=1000`, `nBacktracking=200`, `nAdjacent=10`, 27 polarizations; `A_obj` 4096 × 17,478, 5,826 dipoles | **BLOCKED — no speed claim until §Blocked rung is discharged** |

Rung A's configuration *is* the `native-permanent-magnet-simple` mirror's `native_default`, frozen
in `examples/jax/parity/cases/native_permanent_magnet_simple.py`. Rung B's is **not** any mirror's
`native_default`: the MUSE mirror's is nφ=16
(`examples/jax/parity/cases/native_permanent_magnet_muse.py:33-50`), where the GPU **loses**. Rung B
is therefore a configuration-scoped claim carrying its resolution in its name, as
`flat675-single-stage-coupled-optimization` does; §Scoreboard consequences pre-registers what each
verdict may do to `docs/jax_example_device_assignment.md`. Out of scope: `permanent_magnet_QA` /
relax-and-split (P3.5 — a footprint probe plus a native-only ~32.4 s timing; no matched GPU
number exists); nφ=64 for `pm-simple`; contended-box claims;
batched-instance workloads (that record's scope guard).

## Motivation — diagnostic probe evidence, never campaign evidence

The 2026-08-23 Phase-3 probes (`benchmarks/pm_gpmo_probes.py`, self-labelled
`diagnostic-not-certifying`) motivate this charter and supply **no admissible number**: two timed
solves per leg, no interleaving, no quiet gate, no five-pair rule. Their artifacts sit in-tree under
`docs/receipts/evidence/` (`pm_simple16_*_20260823.json`, `muse_shipped_*_20260823.json`,
`muse64_*_20260823.json`, `pm4stell64_*_20260823.json`), tracked since `fbab4f2b8` — but they are
self-labelled diagnostic and belong to no `docs/receipts/` campaign receipt, so plan clause 7 gives
them no evidentiary standing. Every number below is re-measured or it does not exist.

- **Rung A.** GPU warm 0.0306 s vs swept-native optimum 0.1586 s (OMP=32) → ≈5.2×; numerator and
  denominator are each leg's mean of its two warm samples (GPU 0.03085/0.03034, native
  0.15953/0.15762). Sweep (first warm sample per leg): 1.2097
  (2), 0.7943 (4), 0.3468 (8), 0.2004 (16), 0.1595 (32), **6.0664 (48)** — OMP=48 is the
  all-hardware-threads collapse the device record names, never a usable denominator. Moments bitwise
  (0 ULP, 43,008 elements, one sha both lanes).
- **Rung B.** GPU warm 7.32 s vs swept-native optimum 21.22 s (OMP=32) → ≈2.9×; observed 42.39 (8),
  23.63 (16), 21.22 (32). GPU cold-in-process (8.28–8.80 s) also beats every native configuration
  measured; moments bitwise (0 ULP, 22,590 elements). Device memory 6,409 MiB after solve on timed
  legs, 10,505 MiB on the memory rung (K=100, `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
- **Context that scopes Rung B.** At shipped MUSE scale (nφ=16) the GPU **loses**: 4.70–4.72 s warm
  vs 3.008 s at OMP=32 → 0.64×. The device row's archived 4.05× is stale (it predates the
  frozen-step `lax.cond` skip) but its *direction* survives at that scale: same code, same box,
  different configuration.
- **Blocked member.** `pm4stell-64` timed ≈3.0× (9.51 s vs 28.70 s at OMP=32) and **failed the
  physics check**: endpoints differ (`d2daf391…` vs `db044092…`, max |Δ| 7.96e6).

## Preregistration discipline and instrument freeze

Plan clause 1 (`…backlog…plan.md:190-193`;
`docs/jax_gpu_finitebuild_native_speed_successor_plan.md:7-8,209,247-251`): a **new preregistration
with its own gate derivation**, not an amendment to any prior PM measurement. Amendments are dated,
appended, pre-evidence only, each citing its basis by artifact path and sha256; nothing in
§Fair-native denominator, §Win rule, §Physics gate or §Kill criteria is amendable after any pair has
run.

`benchmarks/pm_gpmo_probes.py` is the instrument — committed at `76f1b5f37`, clean at drafting —
carrying the case SSOT (`CASES`), the history model, the moments compare and the artifact writer;
`benchmarks/probe_conventions.py` supplies identity, ledger, ULP helpers and environment pinning.
**Freeze rule:** the instrument must be committed and its sha256 recorded here under a dated
amendment before the first campaign leg; no edit is permitted while evidence is collected. Required
pre-freeze work, all instrument-side:

1. **Certifying driver mode**, in this file or a thin sibling delegating to it (the
   `benchmarks/nested_ls_a100_banana_omp.py` precedent, 86 lines) — either is acceptable, but the
   choice is frozen here, not discovered mid-campaign. Today one invocation times one leg
   (positional case plus `--lane/--omp/--repeat`, e.g. `pm-simple-16 --lane native --omp 32`)
   and `interleave_schedule` prints as a *suggestion only*
   (`benchmarks/pm_gpmo_probes.py:2429-2434`); the driver must execute the alternation itself in
   fresh processes, publishing the executed order from the ledger — never a planned one.
2. **Quiet gates enforced, not merely available.** `cpu_utilization_delta` and
   `gpu_compute_processes` both exist in `benchmarks/probe_conventions.py`, but the PM probe records
   only the latter. The certifying mode samples CPU busy-ness around every timed leg and discards,
   fail-closed, any leg sharing the box with foreign compute.
3. **Process-wall timer.** The probe publishes in-process solve seconds only; warm persistent-cache
   **process wall** is a second required timer and must be driver-measured.
4. **Frozen scale record for nφ=64.** `ExecutionScale` is a two-value literal
   (`src/simsopt_jax/runtime/execution_scale.py:7`), so Rung B freezes against `CASES["muse-64"]`
   and its grid digests (`A_obj_sha256`, `b_obj_sha256`) rather than a third parity scale — a
   disclosed deviation from P3.6's `_scale_configuration()` route, because a third scale value
   ripples through the whole parity harness.

**Amendment (dated 2026-08-24, pre-freeze; basis:
`docs/receipts/evidence/qa64_jaxgpu_solve_refusal_20260824.log`, sha256
`de004deb63b3524b31739d35a8a86c685bef37eaa30549571b751df6a1e0cc73`, and the P3.5 adjudication in
`docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md`).** The instrument's
`run_jax_relax_split` staging order is repaired — raw grid staged before the host
`rescale_for_opt` shift, the P3.5 double-shift false reject — on operator instruction
(2026-08-24), which lifts the "deliberately not hot-patched; owned by the QA charter" deferral
the instrument's own 2026-08-23 docstring recorded (commit `8c25d4780`). The freeze rule's
window has not opened: the charter is a draft, none of pre-freeze items 1–3 exist yet, and no
campaign leg has run — the 2026-08-23/24 legs are all `grade: diagnostic-not-certifying` under
§Motivation and mint nothing. Post-repair instrument content sha256:
`5ef087a1555cae60b464bca55ab1c4252c313676d4ac112aebbb7d19b6703b0a` (re-pinned, as this rule requires, at freeze time
against the committed instrument). The 2026-08-24 diagnostic legs that validated the repair ran
against this edit while uncommitted and self-disclose it: their artifacts'
`identity.git.status` records the dirty instrument — provenance working as designed, not
evidence contamination, because nothing they produced is campaign evidence.

## Fair-native denominator

Plan clause 3 (`…backlog…plan.md:199-207`;
`docs/jax_gpu_finitebuild_native_speed_successor_plan.md:186-189`; sweep-down correction
`docs/receipts/stage_two_minimal_coupled_route.md:574-583`):

- **Full OMP sweep** `{2,4,8,16,32,48}` per rung, every value measured, the **swept optimum** as
  denominator, shipped default disclosed separately. The probes swept all six for Rung A but only
  `{8,16,32}` for Rung B and the blocked member; completing `{2,4,48}` there is campaign work, and a
  denominator from an incomplete sweep is `NOT_PRODUCED`. Callback and print cost stays charged to
  native, and `--omp` is mandatory on every timed leg — no lane sets threads implicitly.
- **Matched algorithm variant** and **matched K**, read from the instrument's case SSOT, never
  re-typed.
- **Variant-aware history policy, published per lane, never asserted equal.** For
  `ArbVec_backtracking` no setting equalizes the lanes: `gpmo_arbvec_backtracking_solve` has no
  `retain_history` parameter, so its minimum is one recorded row, while `GPMO_ArbVec_backtracking`
  writes twice *outside* its verbose predicate (`src/simsoptpp/permanent_magnet_optimization.cpp`,
  pre-loop and magnet-limit-exit sites). Policy `off` = "the least history this variant can be made
  to keep", and per-lane write counts and byte formulas publish (`grid.history.native.buffer_bytes`
  vs `grid.history.jax.history_bytes`). Note `record_every=None` is the *maximal*-memory JAX setting
  here (full-trace scan, ~1.68 GiB ×2 at MUSE scale), so "no history" is spelled `record_every=K`.
- **Native in-window copies disclosed, not subtracted.** The timed native window includes the GPMO
  wrapper's `A_obj*mmax` scaling, `m_history` rescale loop, prints, and two host copies with no JAX
  counterpart — `contig(A_obj.T)`, a full transposed copy of the scaled response matrix, and the
  `Nnorms` ravel (`src/simsopt/solve/permanent_magnet_optimization.py`, `GPMO`). This inflates the
  native denominator — anti-GPU — and stands because the probe times the wrapper a user actually
  calls (`benchmarks/pm_gpmo_probes.py:2534-2543`).
- **Iteration-semantics disclosure.** `iterations` is a budget K, not an executed count. Native
  `ArbVec_backtracking` breaks at `num_nonzero >= max_nMagnets`; the JAX `lax.scan` has no early
  exit and runs all K, dispatching a cheap-but-nonzero frozen step past the stopping point. **The
  lanes do not do matched work at equal K and this charter does not divide that out**
  (`benchmarks/pm_gpmo_probes.py:296-309`); both publish `iteration_report`, and a rung whose native
  lane exits early publishes the executed prefix beside its ratio. On Rung B and the blocked member
  the asymmetry runs *pro*-native.

## Win rule

Plan clause 2 (`…backlog…plan.md:194-198`;
`docs/jax_gpu_finitebuild_native_speed_successor_plan.md:198-206`), applied **independently per
rung** — no rung inherits its sibling's verdict:

> **Five interleaved pairs, alternating order.** WIN requires median paired
> `native_seconds / gpu_seconds ≥ 1.10` **and every one of the five pairs `> 1.00`**, on both
> `warm_solve_seconds` and warm persistent-cache `process_wall_seconds`. Anything else is
> `CLOSED_BOUNDED_NEGATIVE`. `NOT_PRODUCED` stays what it always was: broken evidence, never a
> verdict.

No dual-anchor clause is adopted: neither rung has an archived anchor of the kind
`docs/jax_gpu_flat675_fused_campaign_plan.md:226-245` binds (the 4.05× MUSE figure is host-local,
stale and a different scale).

## Warm/cold scoping

Plan clause 4 (`…backlog…plan.md:208-211`). Claims are **warm same-process** and **warm
persistent-cache** only; the lever is `JAX_COMPILATION_CACHE_DIR` via `--cache-dir` (absent = cold
lane; the instrument refuses an inherited cache quietly warming a cold leg). Cold is measured and
published for both rungs under the same rule, **never folded into** a warm claim, and may stand
alone only if it independently passes. Cold/warm is symmetric: solve index 0 is `cold_in_process` on
**both** lanes and excluded from `warm_seconds` on both. The two colds differ — in-process first
solve (what the probes measured) is not a fresh process with an empty cache; the campaign publishes
the latter.

## Physics gate

Plan clause 5 (`…backlog…plan.md:212-214`). GPMO is exact greedy arithmetic, so **the gate is
bitwise**: every timed pair's endpoint moments must compare `bitwise_identical: true`, `max_ulp: 0`,
through the instrument's **metadata-gated** `compare_moments`, which refuses two archives
disagreeing on case, nφ, nθ, `downsample`, `iterations`, history policy or the `A_obj`/`b_obj`
digests, so a difference can never be a configuration difference wearing a physics costume
(`benchmarks/pm_gpmo_probes.py:1956-2015`). No tolerance bucket is available here
(`src/simsopt_jax/parity_tolerances.py` governs only non-exact-greedy families), so any non-bitwise
timed endpoint is `NOT_PRODUCED` and the rung joins §Blocked rung.

## Blocked rung — `pm4stell-64` and its adjudication work item

`pm4stell-64` is a **named blocked member**: a speed observation (≈3.0×) with **no admissible
claim**, and none may be minted until the divergence below is root-caused and adjudicated.

Observed 2026-08-23 (diagnostic): the greedy selection forks at **exactly iteration k=201** from
bitwise-identical inputs (both lanes report `A_obj_sha256` `6fd19146…`, `b_obj_sha256` `d87e3be1…`),
each lane is internally bitwise-stable across OMP values and repeats, and the lanes agree bitwise
for k ≤ 200. Re-derived from the archived dumps `docs/receipts/evidence/pm4stell64_fork_k201_native_20260823.npz`
and `docs/receipts/evidence/pm4stell64_fork_k201_jaxgpu_20260823.npz` (both `iterations: 201`,
matched metadata): the k=201 states differ in **10 of 5,826 dipole rows**, 133 non-zero rows native
against 139 JAX.

Two hypotheses, pre-named so neither can be adopted post hoc:

- **H1 — candidate-argmax tie.** An fp tie or reduction-order-sensitive near-tie in the
  27-polarization argmax: native reduces over OpenMP-partitioned rows, JAX on device, and one
  differing argmax cascades through `nAdjacent=10` bookkeeping.
- **H2 — first backtracking pass.** `nBacktracking=200` for this case, so k=201 is the first
  iteration after the first backtracking sweep; a threshold comparison inside that sweep is the
  other reduction-order-sensitive site, and it explains a 10-row jump more naturally than a single
  argmax flip. Discriminant already in hand: `muse-64` runs the same variant at the same
  `nBacktracking=200` and does **not** fork — it differs in `nAdjacent` (1 vs 10) and in
  polarization set.

**Adjudication work item** (must complete before any `pm4stell` rung is chartered): (i) extract both
lanes' iteration-201 candidate scores at the archived state, from the archived dumps, without
re-running the solve; (ii) identify the tied or near-tied candidates and report their score gap in
ULP, which decides H1 against H2; (iii) decide, and record as a dated pre-evidence amendment, either
an **exact-arithmetic tie-break rule** fixed identically in both lanes (preferred — it restores the
bitwise gate) or an **accept-with-bucket physics gate** chartered for this member alone, stating the
bucket, its derivation, and why bitwise is unattainable. Only then may a `pm4stell` rung be added by
amendment — and a tie-break change is a **solver change**, so it also requires parity
re-certification of that mirror before any timing.

**Adjudication record (dated 2026-08-23, later — static phase complete; confirming replay
pre-registered below, pending a quiet box).** Verdict: **the fork site is H2's; the mechanism is a
near-tie broken differently per lane — H1's *kind* of cause, though by FMA contraction rather than
the reduction-order route H1 posited (the summation order is sequential in both lanes); neither
lane is buggy.** The archived dumps are whole-run endpoints of
`iterations: 201` runs (`write_moments`, `benchmarks/pm_gpmo_probes.py:1919-1941` — no
mid-iteration capture exists), and an off-by-one in this section's own framing is corrected here:
the C++ gate is `(k % backtracking) == 0` with no `k >= backtracking` guard
(`src/simsoptpp/permanent_magnet_optimization.cpp:861`), so with `iterations=201` the last executed
iteration k=200 runs the **first non-trivial dewyrming sweep** — the fork is *inside* that sweep,
not one iteration past it. The same correction applies to this section's opening claim that "the
lanes agree bitwise for k ≤ 200": agreement holds through all 201 *placements* (the state entering
the sweep); the end-of-k=200 states differ, because the sweep runs inside k=200. Forensics (from
the archived endpoint dumps — which hold moments only, so placement-identity is *inferred* from
the exactly-antiparallel survivor structure, not read directly): the 201 placements
entering the sweep are identical in both lanes, exonerating H1's argmax site; native removed 34
exactly-antiparallel pairs (201 − 2·34 = 133), JAX 31 (201 − 2·31 = 139); the 10 differing rows are
five single-polarization antiparallel pairs, three of them direct threshold flips and two cascade
consequences. Mechanism: the removal test `min_cos_angle <= cos(threshold_angle)` is an
equality-grade test at this case's `threshold_angle = π` (`cos = −1.0` exactly), and the two lanes
round the 3-term dot differently — the local build compiles the C++ accumulation loop
(`…optimization.cpp:885-888`) with `-O3 -march=native -ffp-contract=fast` (`CMakeLists.txt:59`,
confirmed in `build/cp311-cp311-linux_x86_64/compile_commands.json`), i.e. FMA-contracted, while
the JAX lane's `jnp.sum(moment_j * moment_c)` (`src/simsopt_jax/core/pm_optimization.py:1772`) is
uncontracted. All five deciding pairs straddle −1.0 within 0–2 ULP; sequential-order-with-FMA is
the unique scheme (of 18 tested — session analysis, unarchived; the pre-registered replay below is
the durable check) reproducing every native decision. The `muse-64` discriminant is
resolved and favors this verdict: with `nAdjacent=1` the neighbor scan sees only `Connect(j,0) = j`
itself (self-dot ≈ +1, never ≤ −1), so MUSE's sweep is a structural no-op and *cannot* fork,
regardless of polarization set. Fragility disclosed (same session analysis, unarchived): 10.4% of
all 157,302 antiparallel (dipole, polarization) self-dots flip the `<= −1.0` test between FMA and
non-FMA evaluation.
Line-for-line comparison found **no semantic difference** in the sweep (seed order, neighbor order,
strict-`<` first-wins argmin, cascade bookkeeping all match); two second-order reduction-order
divergences in the residual updates (`…optimization.cpp:903-916` vs `…pm_optimization.py:1706-1708`;
`…optimization.cpp:848-852` vs `:1885`) are noted for later iterations and are not implicated at
k=200. Consequences: (a) "make JAX match native" is not a well-defined target — the CI build
(`-O3 -march=westmere`, `CMakeLists.txt:48`, pre-FMA3) would take the non-FMA branch and agree
with JAX on the three direct pairs, so the native lane is not self-consistent across its own
builds; the work item's **exact-arithmetic tie-break rule is adopted** (at `threshold_angle = π`
the intended predicate is exactly testable as same polarization index ∧ opposite signs; a general
angle needs a correctly-rounded compensated dot with `-ffp-contract=off` pinned on that translation
unit) — a solver change requiring parity re-certification before any timing, per the work item.
Clamping the cosine into [−1, 1] is rejected as a fix: it would move all 157,302 near-boundary
pairs and change the physics on both lanes. (b) **Confirming replay, pre-registered:** rebuild
`simsoptpp` in a separate build dir with `-ffp-contract=off` appended and nothing else changed;
rerun `pm4stell-64` native at `iterations=201`; compare moments to the archived JAX dump. Expected
under this record: the three direct pairs invert (native placed count moves toward 139), ideally
bitwise-identical to the JAX dump; a result byte-identical to the archived *native* dump instead
**refutes** this record and reopens the adjudication.

**Confirming replay — executed and CONFIRMED (dated 2026-08-24).** The pre-registered replay ran
on 2026-08-24: `simsoptpp` rebuilt in a separate cloned build dir with `-ffp-contract=fast`
substituted by `-ffp-contract=off` (verified on the `permanent_magnet_optimization.cpp.o` compile
line), then the `pm4stell-64` native leg rerun at the archived configuration (`iterations=201`,
`--history off`, OMP=8, `--repeat 2`). Result: the contract-off native endpoint is
**bitwise-identical to the archived JAX dump** — `np.array_equal` true, 0 of 5,826 rows differ,
139 placed rows, both in-process solves removing 31 antiparallel pairs where the production build
removes 34 — and it differs from the archived *native* dump in exactly the 10 known rows. This is
the record's ideal outcome, stronger than the minimum "three direct pairs invert": the **entire
fork is FMA contraction**, and the adjudication record above stands confirmed. Artifacts:
`docs/receipts/evidence/pm4stell64_fork_k201_native_ffpoff_20260824.npz` (endpoint),
`docs/receipts/evidence/pm4stell64_native_ffpoff_20260824.json` (whose
`identity.simsoptpp.sha256` names the contract-off binary, `7c560e6b…`, against the production
`41b2ca79…`), `docs/receipts/evidence/pm4stell64_native_ffpoff_20260824.log`. Deviations from the
pre-registration, disclosed: (i) the box was **not quiet** (concurrent campaign legs; run
`nice -n 19`) — admissible because the deliverable is a bitwise endpoint, not a timing, and the
repeats agree; (ii) "nothing else changed" required two build-infrastructure repairs with no
codegen effect on the kernel: the cloned dir's cmake regen edge was deleted (its dependencies
name a deleted ephemeral uv build env, so ninja would otherwise re-run cmake against the
production tree) and two dead include roots were re-pointed (donor pybind11 2.13.6 headers —
binding glue only — and the venv's numpy `_core/include`); (iii) a **first replay attempt the
same morning is VOID and adjudicates nothing**: its parent-process module preload did not survive
`_reexec_native_child`'s scrubbed re-exec, so the leg ran the production binary while labeled
ffp-off — proven by its own `identity.simsoptpp.sha256` = `41b2ca79…` — and trivially matched the
native dump. Its ledger line (2026-08-24T08:48Z) stands as an executed-leg fact; its artifacts
were removed from the evidence tree. The working mechanism for any future replay is a
`sitecustomize.py` preload on `PYTHONPATH` (which `pinned_environment` deliberately preserves
into the child), verified after the fact by the artifact's identity block. Consequence unchanged:
the exact-arithmetic predicate repair remains chartered (solver change → parity re-certification
before any timing); this replay mints no `pm4stell` timing rung.

**Repair — implemented and lane-parity re-certified (dated 2026-08-24, same day).** The adopted
exact-arithmetic predicate is now in both lanes: at ``thresh_angle == pi`` (``cos_thresh_angle ==
-1.0``, an equality-grade test) the dewyrming removal decision and partner selection use exact
componentwise negation — first adjacent placed neighbor with ``x_cj == -x_j`` in all three
components, FP equality so ``+-0.0`` compares equal — instead of the rounded 3-term dot
(``src/simsoptpp/permanent_magnet_optimization.cpp`` sweep; ``src/simsopt_jax/core/pm_optimization.py::_gpmo_arbvec_remove_pairs``,
gated at trace time off the static spec angle). First-qualifying-neighbor-wins is the exact
limit of the strict-``<`` first-wins argmin the general-angle path keeps unchanged, and the
integer predicate the plain ``GPMO_backtracking`` variant always used — the repair restores the
family's own convention. Re-certification at the fork scale: a rebuilt native kernel **with FMA
contraction left ON** (``-ffp-contract=fast``, sha ``95190afa…``) and the JAX GPU lane (RTX
5090) produced **bitwise-identical `pm4stell-64` endpoints** at ``iterations=201``/history off —
``np.array_equal`` true, 163 placed on both (`docs/receipts/evidence/pm4stell64_fork_k201_{native,jaxgpu}_predicate_20260824.npz`
+ paired JSONs/logs; identity blocks name the binaries). The bitwise cross-lane gate this
section required is restored, build-scheme-independently. Both endpoints differ from both
archived pre-repair dumps, as the exact semantics require: the exact predicate removes the 19
exactly-antiparallel pairs (201 − 2·19 = 163) while the FP predicates also removed rounded
near-ties (34 native-FMA / 31 uncontracted). Committed regression pins
(`tests/jax/core/test_pm_optimization_jax_item25.py::TestGPMOArbVecBacktracking::test_thresh_pi_removal_is_exact_componentwise_negation`):
exact pair removed, one-ULP-off pair kept (the pre-repair FP predicate removed it on every
build), mixed ``+-0.0`` negation removed, general ``0.9 pi`` FP path unchanged with C++ oracle
parity. Two review-hardening notes: every shipped caller of the JAX step — the jitted solve entry,
the live-loop workflow (`src/simsopt_jax/core/pm_workflow.py`), and the step-level test
harnesses — now derives BOTH the exact-mode gate and the general-angle threshold from the host
libm cosine (``math.cos``), the same libm the C++ twin's ``std::cos`` uses, so one cosine
implementation owns the branch decision and the FP threshold in both lanes and a
device-rounded ``jnp.cos`` can no longer split them at a near-``pi`` angle (the audit caught
the live-loop path still on ``jnp.cos`` and it was closed before landing; folding the
derivation into the step itself — dropping the ``cos_thresh_angle`` parameter so an
inconsistent pair is unrepresentable — is a named pre-freeze follow-up). Note the gate is
``cos(thresh_angle) == -1.0``, not a literal ``pi`` comparison: it deliberately captures every
angle whose cosine rounds to exactly ``-1.0`` (a ~1e-8 band around ``pi``), throughout which
the FP test is equality-grade and the exact predicate is the correct semantics. And the exact
predicate is precisely the moment-cancellation test (``m_j == -m_c``), which for
the unit polarization vectors every in-repo builder produces coincides with antiparallelism —
for hypothetical non-unit ``pol_vectors`` the old dot-threshold was already angle-incorrect in
both directions, so no valid behavior is lost. Disclosures: (i) the shipped prebuilt ``build/cp311-cp311-linux_x86_64`` binary is NOT
rebuilt by this repair — committed C++ source now leads it; the full suite (item25 66, item28
51, mirror-parity muse/pm4stell, pm_optimization, pm_workflow_jax — all green 2026-08-24)
remains valid against the old binary because no committed test constructs a reachable near-tie,
and the one new assertion that forks (the one-ULP-off case) is deliberately JAX-only until a
rebuild ships. The committed re-cert receipts pin the leg-time sources in their
``identity.git.changed_file_sha256`` blocks; the post-leg C++ delta is a comment trim only,
and this is sha-proven, not asserted — replacing the committed 5-line comment inside the
``if (cos_thresh_angle == -1.0)`` body with the pre-trim block below reproduces the leg-time
file byte-for-byte (``6fb2bba505a629a13eba6392b9899cc8106dbdada4cfdf8263a78080a5cb64d6``,
independently re-derived by the review's auditor from a pre-trim diff artifact):

```cpp
                    // thresh_angle == pi: the removal test is equality-grade
                    // (cos <= -1 can only mean exactly antiparallel), so it is
                    // evaluated exactly -- componentwise negation, no dot
                    // product -- because the rounded 3-term dot straddles -1.0
                    // differently under FMA contraction than under plain
                    // rounding, forking the removal set between builds of this
                    // same file. First qualifying neighbor wins, matching the
                    // exact-arithmetic limit of the strict-< argmin below and
                    // the integer predicate GPMO_backtracking already uses.
```

(ii) the re-cert legs ran ``nice -n 19`` on a non-quiet box — admissible for
bitwise endpoints, and **no timing rung is minted**: fair-timing legs remain chartered under
this plan's frozen-instrument rules, now unblocked.

**QA companion (same commit): the P3.5 instrument-side staging fix is applied** — on operator instruction (2026-08-24), lifting the deliberate 2026-08-23 deferral to the future QA charter (see the dated instrument amendment under §Preregistration discipline).
``benchmarks/pm_gpmo_probes.py::run_jax_relax_split`` now stages the RAW grid before the host
``rescale_for_opt`` shift (the ordering is the contract; docstring rewritten), so the explicit
alpha — still derived exactly as the native lane derives its step — passes the ``_mwpgp_spec``
validator it used to false-trip. The historically-refusing qa-64 jax-gpu solve leg now
completes: rc=0, both continuation solves finite over all 29,286 dipoles
(`docs/receipts/evidence/qa64_jaxgpu_solve_20260824.json` + `.log`, endpoint
`qa64_rs_jaxgpu_20260824.npz`; diagnostic seconds 5.34 cold / 4.93 repeat_retrace — no
denominator, no claim). Ordering-contract regression pinned at
`tests/solve/test_permanent_magnet_optimization_jax_item28.py::test_relax_and_split_jax_staging_order_contract_single_shift`
(fixed order accepted and equal to the default-step formula; buggy order refused with the
archived operand signature). One transient during validation, disclosed: the first fixed-leg
attempt crashed with an async ``CUDA_ERROR_ILLEGAL_ADDRESS`` at the post-solve sync right
after the pm4stell GPU leg vacated the card; the clean-GPU retry ran green with zero CUDA
errors, and the crash is recorded as environment-transient, not a defect of the fix (rerun on
a quiet card before treating any recurrence as real).

## Provenance, ledger, quiet gates

Plan clause 6 (`…backlog…plan.md:215-217`). Every leg writes a `probe_conventions.runtime_identity`
block — commit + dirty sha, observed OMP, JAX platform/x64/devices, unfiltered `nvidia-smi`
processes plus `own_pid`, simsoptpp build, library versions — and appends to
`docs/receipts/evidence/probe_leg_ledger.jsonl` in **executed** order, from a scrubbed-then-pinned
fp64 environment (`pinned_environment`). Cross-leg conformance is fail-closed: two legs of a rung
disagreeing on commit, pins or grid digests invalidate its pairs.

## Kill criteria (fail-closed, none amendable post-evidence)

- Any timed pair's endpoint not bitwise identical → `NOT_PRODUCED`; the rung joins §Blocked rung.
- Median paired ratio `< 1.10` on either required timer, or any single pair `≤ 1.00` → that rung
  closes `CLOSED_BOUNDED_NEGATIVE`.
- An incomplete OMP sweep, an unset `--omp` on a timed leg, a denominator taken from the
  pathological all-threads configuration, or an instrument sha changing between freeze and last pair
  → `NOT_PRODUCED`.
- Foreign CPU or GPU compute during a timed leg → leg discarded and re-run; three discards on a rung
  → `NOT_PRODUCED`.
- Rung B additionally: device memory exceeding the card at the frozen configuration →
  `NOT_PRODUCED`, with the measured footprint published.

## Evidence layout, receipt, scoreboard amendment

Plan clause 7 (`…backlog…plan.md:218-222`). Terminal receipt at
`docs/receipts/pm_gpmo_native_gpu.md`, tracked; evidence bundle at
`docs/receipts/evidence/pm_gpmo_native_gpu/`, tracked, holding every leg artifact, the ledger slice,
both moments archives per pair and the `compare_moments` reports. Both timers' medians, all pair
ratios, the full OMP sweep and the cold numbers publish **regardless of verdict**.

`tests/test_jax_example_device_assignment.py` enforces the rest: a `gpu` row must cite a path `git
ls-files` reports as a tracked regular file under `docs/receipts/` (a host-local directory, an
untracked file, or a directory will not satisfy it), and the same rule gates `measured-certified`.
Amend by **appending a dated log row and editing the table row it refers to in the same commit**;
never rewrite a log entry (`docs/jax_example_device_assignment.md`, §Scope note and amendment
procedure).

## Scoreboard consequences, pre-registered per rung

- **Rung A WIN** → `native-permanent-magnet-simple` moves `gpu/measured-diagnostic` (its state
  since the 2026-08-23 P3.2 amendment) → `gpu/measured-certified`, citing the receipt; it is the one
  row a rung here can hold at `gpu`, because Rung A's configuration *is* that mirror's
  `native_default`. **CLOSE** → the row leaves `gpu`: `cpu/measured-certified` if the evidence
  establishes native faster, else back to `unmeasured` with the diagnostic 5.2× superseded in the
  cell by this campaign's measured number.
- **Rung B WIN** → `native-permanent-magnet-muse` **stays `cpu`**: its `native_default` is nφ=16,
  where the GPU loses, so a `gpu` move on a nφ=64 result would misstate the record's own semantics
  ("where do I launch *this example*"). The amendment supersedes the stale host-local 4.05× with the
  matched-work shipped-scale number and records the nφ=64 claim as a scoped, named configuration
  result. A `gpu` row for that configuration first needs it to become a reachable example mode (the
  `--bundle` precedent), not chartered here. **CLOSE** → mechanism-text amendment only.
- **`pm4stell-64`** → `native-permanent-magnet-pm4stell` stays `unmeasured` under every outcome
  here; only the adjudication work item can change that.
