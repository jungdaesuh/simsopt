# Projected-route example promotion — diagnostic receipt

> **DIAGNOSTIC, NOT CERTIFYING.** This receipt records one measurement campaign
> run on one workstation on **2026-08-17 UTC** (2026-08-16 local, US/Eastern).
> **Every date in this document is UTC**, matching the `wallclock_utc` stamps
> in the leg JSONs; the campaign directory's `-20260816` suffix and the
> "2026-08-16 pre-commit audit" are local-time names for the same session and
> are left as the identifiers they are. It is not a certification: no artifact
> here is sealed, no digest here is frozen into a gate, and the campaign
> artifacts live outside this repository. Where it disagrees with a sealed
> record, this document is the newer *measurement* and the sealed record is the
> older *certificate*; §8 states exactly what that does and does not settle.

| | |
|---|---|
| Question | Does `examples/jax/3_Advanced/single_stage_boozer_vacuum_projected_route.py` reach native-equivalent physics quality at the example level while running faster on GPU? |
| Verdict | **BOUNDED NEGATIVE on speed** against the native mirror — the GPU example is **5.07x** slower at matched process wall and **8.37x** slower to matched endpoint quality, both **pooling** the 8 warm+cold GPU legs (pooled median 262.124 s); warm-only, 5.14x and 8.48x. **Equivalence PASSES**: all seven pre-registered physics gates hold on two independent endpoint pairs, both recomputed through the native evaluator. |
| Secondary result | For the script's own device choice the GPU wins decisively: on the CPU backend the example **does not reach its objective target at all** (3/3 attempts `LINE_SEARCH_COLLAPSE`, published `retry_exhausted`), and one iteration costs **8.33x** more there. Device-assignment row moved `unmeasured` → `gpu`. |
| Repository | `pr/jax-port-squashed` @ `dd5e3113b8a549ff3f5ae46ea0b2b957e9cabf21`, tree clean at every leg (stamped per leg) |
| Interpreter | CPython 3.11.15, jax 0.10.0 (CUDA), numpy 2.4.6, scipy 1.17.1, `.venv-qn-gpu` |
| Box | RTX 5090 + 64-core host, **shared with a foreign workload for the whole campaign** (§7) |
| Artifacts | `~/simsopt-campaigns/projected-route-example-promotion-20260816/` — 41 leg JSONs (37 with a sampled contention record), 15 harness files, `artifacts/receipt_numbers.json` binding **169** numerals to a file and a pointer, and `artifacts/receipt_numeral_audit.json` classifying **every** numeric token in this receipt, integers included, with `unmatched_after_audit` empty (see §10 for the two tokenizer defects that made an earlier coverage claim narrower than it sounded) |

---

## 1. What was compared

Two example scripts, both at `native_default` scale, both run as whole
processes under a supervisor that measures Popen-to-exit wall:

* **native** — `examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`.
  Nested two-stage: scipy BFGS over 461 coil DOFs, one native Boozer Newton
  solve per objective evaluation. Budget `NATIVE_ITERATIONS = 1000`.
* **projected route** — `examples/jax/3_Advanced/single_stage_boozer_vacuum_projected_route.py`.
  Coupled full-space: 716 joint coordinates (461 coil + 253 surface + iota + G),
  255 exact equality constraints, projected-Lagrangian Newton-CG with exact
  retraction. Budget 700 iterations, stops on `OBJECTIVE_TARGET_REACHED`.

They are the same physics problem. Proven, not assumed:

* the two bootstraps produce **bitwise identical** 461-coordinate coil DOF
  vectors (`bootstrap_coil_dofs_bitwise_identical = True`), and their DOF name
  suffixes match element-for-element (the raw names differ only by Optimizable
  instance counter — `Current2:x0` vs `Current5:x0` — because the objective
  graph builds its own `BiotSavart`);
* the two examples publish initial objectives agreeing to **1.5889e-14**
  relative;
* the campaign's own native replica (`harness/native_evaluator.py`, copied
  line-for-line from the example) reproduces the example's published initial
  objective **bitwise** (`replica_relative_difference = 0.0`).

**Process wall is the headline boundary** because it is the only boundary the
two scripts share. Every other boundary is published too (§3), and the GPU
loses at all of them.

---

## 2. The declared band and its pre-registered derivation

The band was registered in `harness/bands.py`
(sha256 `57f1d782ff177278cbfbc44b536ee57996d7bef1a3787268373c125f56291204`,
sealed 2026-08-17T01:14:01Z) **before** any full-scale projected-route endpoint
or timing statistic was read.

> **Timing disclosure, stated plainly.** `harness/bands.py` postdates the
> completed probe artifact: its mtime is **21.483 s after**
> `legs/probe-projected-warm.json` was written, and that leg JSON was already
> complete, with its endpoint published, when the band file landed. An earlier
> revision of this paragraph said the probe was "still being supervised"; that
> was wrong and is withdrawn. **File ordering therefore does not establish the
> no-peeking property.** What it rests on is the authorship claim — the band
> file's contents were written without reading the probe, and the probe was
> first read after the band file was sealed by digest — which is exactly the
> kind of claim a reader should discount. Two things do not depend on it: the
> primary band is not this campaign's number at all (it is reused verbatim from
> an in-repo case file predating this campaign), and §2's threading-fork
> measurement re-earns it independently.

**The primary band is not a new number.** It is
`examples/jax/parity/cases/native_single_stage_boozer_vacuum.py ::
NATIVE_DEFAULT_QUALITY_BAND`, reused verbatim: **final objective ≤ 1.0e-07**.
Its own recorded derivation is the 2026-08-14 three-lane `native_default` run,
in which every lane ended budget-exhausted at 1000 iterations with final
objective 4.3972e-08 (native-cpu), 4.5074e-08 (jax-cpu) and 4.5614e-08
(jax-gpu) from 8.4442e-05; the band sits one decade above the worst of those.
Inventing a second floor would have made the floor a free parameter.

**This campaign independently re-earned that band.** The native lane does not
agree with *itself* to better than the band across thread counts: five OpenMP
thread counts produced **5 bitwise-distinct endpoints** with final objectives
`[4.1925e-08, 4.3059e-08, 4.4424e-08, 4.5246e-08, 4.5285e-08]` (ascending
sorted), a **8.01%** relative spread — while being bitwise reproducible
*within* a thread count (7/7 legs at OMP=8 identical). A final-value equality
gate would fail the native example against itself.

Every other band is an arithmetic consequence or an explicitly justified
margin; none is free. See `artifacts/equivalence_final.json → bands` for the
full text of each derivation.

---

## 3. Scoreboard

Ascending-sorted walls, seconds, supervised Popen-to-exit. `n` counts completed
legs; failures are recorded in §7 and excluded, never dropped silently.

| Configuration | n | median | min | max | vs best native (median) |
|---|---|---|---|---|---|
| native OMP=2 | 1 | 72.017 | 72.017 | 72.017 | — |
| native OMP=4 | 2 | 56.473 | 54.920 | 58.025 | — |
| **native OMP=8 (best)** | **7** | **51.668** | **49.897** | **64.348** | **1.000** |
| native OMP=16 | 6 | 57.618 | 55.311 | 75.832 | — |
| native OMP=32 | 1 | 253.240 | 253.240 | 253.240 | — |
| native, `OMP_NUM_THREADS` unset (shipped default) | 0 completed | — | — | — | **> 28.78x**, terminated at 1852.453 s while still running |
| projected route, GPU, warm cache | 6 | 265.642 | 235.246 | 330.143 | **5.141** |
| projected route, GPU, cold cache | 2 | 232.711 | 221.649 | 243.773 | **4.504** |
| projected route, CPU backend | 0 completed | — | — | — | did not reach its target (§5) |

**Every boundary, not only the flattering one.** The native denominator is
always its *full process wall* — the strictest reading available to it. **Every
row below POOLS the warm and cold GPU legs**: the GPU numerator is the 8-leg
pooled statistic, whose **median is 262.124 s** (min 221.649 s, and it appears
nowhere in the table above, which is split by cache state). Pooling is
charitable to the GPU, because both cold legs are faster than five of the six
warm legs; the warm-only ratio is 5.141 (§3's table) against the pooled 5.073.

| Boundary (pooled 8 GPU legs, median 262.124 s) | ratio (GPU ÷ native) |
|---|---|
| **pooled** GPU wall median ÷ native wall median | **5.073** |
| **pooled** GPU wall **minimum** ÷ native wall median | 4.290 |
| **pooled** GPU **engine** (compile+solve) median ÷ native wall median | 4.469 |
| **pooled** GPU **engine minimum** ÷ native wall median — the most charitable reading available | **3.266** |
| **pooled** GPU wall median ÷ native **time-to-band** (§4) | **8.365** |

The warm-only equivalents, for a reader recomputing from §3's table: 5.141 at
matched process wall and 8.478 to matched endpoint. Nothing in the verdict turns
on which is quoted; both are stated so neither can be mistaken for the other.

**Contention-matched interleaved A/B.** Because the margin needed to survive a
shared box, each round ran the lanes minutes apart under the same foreign load.
Per-round GPU ÷ fastest-native-in-that-round, ascending sorted:
`[4.918, 5.036, 5.131, 5.319, 5.356]`, median **5.131**. The matched statistic
and the pooled statistic agree.

**The thread sweep matters and the old rule stays fallen.** The optimum is
OMP=8. OMP=32 is 4.90x worse than OMP=8, and the shipped example — which pins
nothing, so 64 threads on this box — had not finished after 1852.453 s
(> 28.78x the OMP=8 leg from the same round) when the campaign terminated it.
Anyone timing this native example without pinning threads is measuring the
OpenMP collapse, not the example.

---

## 4. Matched endpoint: native time-to-band

The two examples stop for different reasons — the projected route stops the
moment it reaches its objective target (399 of 700 allowed iterations, in every
leg), while the native example has no such stop and always spends all 1000
iterations. Charging native for iterations it did not need would flatter the
GPU, so native's own time-to-quality was measured on a budget ladder at OMP=8:

| budget | wall (s) | final objective | inside band? |
|---|---|---|---|
| 100 | 7.820 | 5.8751e-07 | no |
| 200 | 13.077 | 2.9395e-07 | no |
| 300 | 18.596 | 1.9138e-07 | no |
| 400 | 24.444 | 1.4707e-07 | no |
| 500 | 28.708 | 1.0680e-07 | no |
| **600** | **31.335** | **8.1005e-08** | **yes** |

Native enters the band at 600 iterations in **31.335 s**. The grid is coarse
(100-iteration steps), so this is an **upper bound** on native's time-to-band:
the true crossing lies between the 500-step and 600-step legs. Against it the
GPU example is **8.37x** slower (median wall) and 7.07x slower at its own
fastest leg.

**Why the reformulation loses here.** The coupled route needs **2.51x fewer**
iterations (399 vs 1000) but each costs **11.20x more** (0.5788 s per projected
iteration at the engine median vs 0.05167 s per native outer iteration; native
spends 1239 objective evaluations, 0.04170 s each, each containing an inner
Boozer Newton solve). 2.51 ÷ 11.20 is the loss.

---

## 5. The same script on its own CPU backend

This is the device question the assignment record actually asks, and it is not
the native-mirror question.

* **CPU backend (`jax_cpu_fast`, fp64, x64, full budget):** the example spends
  its entire three-attempt protocol and **never reaches the objective target**.
  **All three attempts end `LINE_SEARCH_COLLAPSE`; the reported (final) attempt
  ran 74 iterations.** The first two attempts' iteration counts are *not*
  recoverable from the artifact — the example publishes
  `iterations_run = len(reported.run.iterations)` for `reported = attempts[-1]`
  only — so "all three collapsed at iteration 74" would be an inference the
  record cannot support, and is not made here. The run publishes
  `protocol_verdict = retry_exhausted`, `status = failed`, final objective
  1.3765e-06 — 13.8x outside the band and 30.7x its own objective target —
  after 1294.436 s.
* **GPU:** reaches `OBJECTIVE_TARGET_REACHED` at iteration 399 in **8 of 8**
  legs, with a **bitwise identical** 716-coordinate endpoint every time (one
  distinct solution digest across 5 warm interleaved legs, 2 cold legs and 1
  probe) and one distinct final objective.
* **Matched-budget per-iteration cost**, both lanes on the identical truncated
  protocol (`--max-steps 25` then `50`, warm cache, fixed costs differenced
  away): GPU **0.4536 s/iteration**, CPU backend **3.7792 s/iteration** —
  the CPU backend is **8.33x** more expensive per iteration. **This figure is a
  mild upper bound on the CPU lane's disadvantage**: both matched CPU legs ran
  with `OMP_NUM_THREADS = 16`, and the one uncapped datapoint (25 steps, no
  thread pinning) solved in 95.436 s against the capped 97.490 s — about 2%
  faster. No uncapped 50-step leg exists, so the slope could not be recomputed
  uncapped; on the single point available the cap is worth ~2%, not the ~8x
  that would be needed to change the reading.

So the GPU is not merely the faster device for this script; it is the only
device this campaign measured on which the script's own success criterion is
met. That is why the device-assignment row moves to `gpu` even though §3 is a
bounded negative against the native mirror.

---

## 6. Equivalence: seven physics gates, both endpoints through the native evaluator

Equivalence is never mediated by either lane's own evaluator. Both endpoints
are recomputed in native SIMSOPT — C++ Biot-Savart, native `BoozerSurface`
Newton — by `harness/native_evaluator.py`. The symmetric measurement supplies
**only the coil degrees of freedom** and lets the native inner Newton determine
the surface from the bootstrap warm start, exactly as the native lane does at
every one of its own evaluations.

Run on two independent endpoint pairs (`ab-r6-native-omp8` × `ab-r6-projected-gpuwarm`,
and `p2-native-omp8-a` × `p2-projected-cold-b`). Both returned
`equivalence-gates-pass`; the figures below are identical across the two, which
follows from the bitwise-identical GPU endpoint and the bitwise-reproducible
OMP=8 native endpoint.

| Gate | Statement | Band | Native | Projected | Pass |
|---|---|---|---|---|---|
| **G1** objective band | endpoint objective through the native evaluator | ≤ 1.0e-07 | 4.30588e-08 (43.1% of band) | 4.47605e-08 (44.8% of band) | ✅ |
| **G2** feasibility | ‖b‖₂ at each lane's own published state, `b = [boozer_residual[mask], volume − target]` (255 components) | ≤ 1.0e-10 | 9.1340e-14 | 9.4268e-12 (9.4% of band) | ✅ |
| **G3** re-projection invariance | objective shift when the route's state is driven to the native solver's own 1e-13 | ≤ 1.0e-09 | — | **1.0794e-19** | ✅ |
| **G4** volume label | fixed-volume equality defect | ≤ 1.0e-10 | −6.11e-16 | 5.00e-16 | ✅ |
| **G5** finiteness | every recomputed observable finite | — | all finite | all finite | ✅ |
| **G6** total coil length | one-sided length bound with margin | ≤ 21.0101 m (target × 1.001) | 20.989157 m | 20.989148 m | ✅ |
| **G7** per-term bands | every native objective term ≤ the band | ≤ 1.0e-07 | max term 4.306e-08 | max term 4.476e-08 | ✅ |

**Not gated, reported only.** Coil-shape proximity is never a gate — the
2026-08-16 stage-two campaign measured 0.154 m coil differences inside a single
flat valley with field-level equivalence intact. For the record, the two
endpoints' 461-coordinate DOF vectors differ by 2.077% in relative L2.

**A physics result worth naming.** Running the native inner Boozer solve from
the *bootstrap* surface at the projected route's coils lands on the route's own
manifold point to **7.83e-14** in surface-DOF L2. The coupled formulation's
surface is the surface the nested formulation would have computed; the two
routes are on the same branch, not merely at similar objective values.

---

## 6.1 The gate-design finding

**A strict reuse of the native inner solve's stopping rule would have
false-rejected this route by 94x, and the rejection would have been an
artifact.**

The native inner Newton declares success at `‖b‖₂ ≤ newton_tol = 1e-13`. The
projected route's published endpoint sits at `‖b‖₂ = 9.4268e-12` — 94 times
outside that number. A gate that simply reused `1e-13` would have failed the
route.

G2's band is `1e-10`, with the margin stated in three parts rather than
asserted:

1. `1e-13` is a **stopping rule on an inner subproblem**, not a physics
   requirement. Reusing a strict solver bound as a cross-formulation acceptance
   gate is the recorded false-reject pattern — on 2026-07-10 a donor that
   converged at κ = 40.0076 was rejected by a κ ≤ 40 gate for 1.9e-4 relative.
2. The band is **tighter than the route's own contract, not looser**. The route
   enforces `‖b‖∞ ≤ 1e-10`, which over 255 components admits `‖b‖₂` up to
   √255 × 1e-10 = 1.5969e-09. G2 at 1e-10 is **15.97x stricter** than the
   lane's own published promise.
3. The margin's cost is **measured, not assumed**. G3 drives the route's own
   published state down to the native solver's own 1e-13 and reads the native
   objective before and after. The objective moves by **1.0794e-19** — one part
   in 4.1e11 of the endpoint value, and 1.08e-12 of the band. The feasibility
   the margin admits is physically inert at the resolution the band judges.

The ruling this supports, for reuse: **when a reformulated lane is judged
against a nested lane, the inner solver's stopping tolerance is not an
admissible acceptance gate; the admissible gate is the physical equality
residual with a stated margin, and the margin must be discharged by measuring
what re-projection to the strict tolerance costs the gated observable.**

---

## 7. Contention, honestly

**The box was never quiet.** A foreign workload — `scripts/run_trial.sh`,
rooted at `/home/jungdaesuh/code/fusion/fusion_equilibrium_challenge`, relaunching
itself in a loop with a fresh pid each time — held the GPU for the entire
campaign and, when its workers were up, **3.90–16.88 host cores**. That range is
the min and max of the 17 *nonzero* `foreign_cpu_total_pcpu` readings among the
82 before/after box-state observations; the other 65 read 0.0% because the
foreign campaign relaunches in a loop and was between trials at that instant, so
the figure is a range over the observations that caught it running, not a duty
cycle. It is a session observation, not this campaign's process, and it was
never killed.

**Contention-record coverage: 37 of 41 legs, not all of them.** The whole-leg
sampler (2 s period; mean and median device utilization, mean and max 1-minute
load, and the fraction of samples in which a non-baseline, non-self pid held the
device) was added to `run_leg.py` *after* the plumbing and first probe legs had
already run. **Four legs carry no sampled record** — `plumb-native`,
`plumb-projected`, `probe-native-omp16`, `probe-projected-warm` — and hold only
the before/after box-state pair. `scoreboard.json` encodes their absence as a
`-1.0` sentinel in the affected configurations' contention lists.

Two of those four feed published statistics, and this is stated rather than
buried:

* `probe-projected-warm` **is the GPU warm minimum, 235.246 s** in §3's table,
  and is one of the six legs behind the warm median 265.642 s;
* `probe-native-omp16` is one of the six legs behind the OMP=16 median
  57.618 s (it is not that configuration's minimum).

Neither is load-bearing for the verdict: the best-native configuration
(OMP=8, 7 legs) and the cold GPU legs are fully sampled, and the
contention-matched per-round ratios are computed only from `ab-` legs, all of
which are sampled. But the coverage is 37 of 41, and any statement about "every
leg" would have been false.

`harness/boxstate.py` classifies by an **exact pid allowlist captured at
campaign start plus self-pid exclusion** — never by substring match on a process
name, which is the disclosed 2026-08-16 defect that let four batched legs share
a device undetected. The foreign pid present at campaign start was deliberately
**not** baselined.

Measured over the 37 sampled legs: foreign GPU process present in **100%** of
samples on every timing leg but one (82.7%); mean device utilization 93.9–98.6%
during the five warm interleaved GPU legs and 61.0–66.8% during the two cold
legs; 1-minute load mean 6.4–76.3.

**Which way this biases the result — NOT established by this campaign.** An
earlier revision of this receipt asserted that contention penalizes the native
lane more, citing 50–228x native OpenMP degradation. That figure is
**[host-local] out-of-workload context** — it comes from GSCO/GPMO greedy loops
at loads around 240, not from this workload — and this campaign's own legs do
not support the inference. Measured here (`artifacts/contention_bias.json`):

| Lane | wall spread (max/min) | Pearson(1-min load, wall) |
|---|---|---|
| native OMP=8, 7 legs | 1.290 | 0.825 |
| projected route GPU, 7 legs | 1.489 | 0.913 |

The GPU lane's wall is *more* dispersed and *more* load-correlated than the
native lane's, not less. The per-round ratio's correlation with load is weak and
not sign-stable in the regressor: −0.179 against the GPU leg's own mean load,
−0.019 against the round's native legs' mean load. **The direction of the bias
is therefore unestablished**, and no part of the verdict rests on it — the
margin is 5.07x pooled and 4.92–5.36x in every contention-matched round, which
is far outside any dispersion measured here.

**Recorded failures — kept as failures, never dropped:**

| Leg | What happened |
|---|---|
| `ab-r1-native-ompunset` | Still running past **1852.453 s** at OMP unset (64 threads, 1-minute load mean 76.3); SIGKILLed by the campaign at its schedule limit. Recorded in `artifacts/terminated_legs.json` as a **lower bound**, never as a measurement. Excluded from all statistics. |
| `p2-projected-cpu-full` | The example itself published `status = failed` / `retry_exhausted` (§5). Observables complete and analysed; its wall prices three abandoned attempts, so it never enters a wall statistic. |

**A shipped-example note worth a ledger entry.** The native example *always*
exits 1 at `native_default`: `scientific_success_for_scale` requires
`certificate.success`, and scipy BFGS always stops on its iteration limit
(`OUTER_GRADIENT_TOLERANCE = 1e-15` is unreachable). Every native leg here is
**budget-exhausted, not failed** — it published a complete, in-band endpoint —
and the exit code cannot be used as a health signal at that scale.

---

## 8. Relationship to the sealed certification — and one anomaly

`docs/single_stage_jax_gpu_projected_route_certification_plan.md` certifies a
**2.304x** engine-boundary win, and **1.260x** at its strictest supervised
boundary, against a frozen native bar of **287.304218 s**. Nothing in this
campaign contradicts the arithmetic of that certificate, and this campaign did
not re-run its lane.

What this campaign does show is a **large unexplained gap between the bar and a
fresh run of the native example**:

* a fresh run of the native *example* **at this repository's HEAD**, at the same
  budget and on the same problem, completes its **whole process** in
  **49.897–64.348 s** at its measured thread optimum — **4.46–5.76x** under the
  287.30 s bar. Verified problem identity on *this* campaign's side, read from
  the executed sources: NCSX via `get_data("ncsx")`, `mpol = ntor = 6`,
  `sDIM = 20`, inner `newton_tol = 1e-13`, outer budget
  `NATIVE_ITERATIONS = 1000`, outer `gtol = OUTER_GRADIENT_TOLERANCE = 1e-15`;
* the campaign has one mechanism that spans that gap in the right direction:
  the shipped example pins no threads, and unpinned on this box the identical
  run had not finished after **1852.453 s**.

> **The claim "at the same commit" has been withdrawn.** An earlier revision of
> this section said the two runs were at the same commit. They were not: this
> campaign's legs all stamp `dd5e3113b` with a clean tree, while the bar run's
> own preserved receipt self-describes as `repository_dirty: true` at a
> different commit. What the two share is the workload definition, not the
> checkout.

**The boundary mismatch cannot explain the gap.** The bar is an *interior*
figure: `src/simsopt/optimization_trajectory.py` starts a `perf_counter` at
`OptimizationTrajectoryRecorder` construction (`:137`) and every recorded row's
wall is measured from there (`:164`, `:180`), so the bar runs from recorder
construction to the 1000th scipy-BFGS callback and excludes process spawn,
imports, the NCSX build, the initial Boozer Newton solve and the final
evaluation. This receipt's native numbers are whole-process walls, so they
include all of that — the comparison is already stacked *against* this
campaign's side. The 2026-08-16 pre-commit audit closed the residual by
deriving, from the bar run's own artifacts, an upper bound of **≤ 289.90 s** on
that child's whole process wall: a boundary gap of **≤ 2.60 s**, **≤ 0.91%**.
**This strengthens the finding** — the 4.46–5.76x anomaly survives converting
both sides to the same, strictest boundary, so it is not a boundary artifact and
must be a property of the runs themselves.

The mechanism remains a hypothesis: the bar is a benchmarks-path run at an
unrecorded thread setting on a box whose contention is unknown, and its own
preserved receipt self-describes as `authoritative: false`,
`repository_dirty: true`, `normalized_status: budget_exhausted`. **Ledger item:
the 287.30 s bar should be re-timed under a stated `OMP_NUM_THREADS` before it
is used again as a comparison denominator.** Until then, the honest statement
is that the sealed claim is sound against *its* bar, and that *this example*
against a *thread-optimised fresh native example* loses by 5.07x (pooled).

**Other anomalies.**

1. **Cold beat warm.** The two cold-cache GPU legs (221.649 s, 243.773 s) are
   faster than five of the six warm legs (235.246–330.143 s) — because they ran
   at 61.0–66.8% mean device utilization while the warm legs ran at 93.9–98.6%.
   The cache's own effect is isolated in the compile term alone: **3.82–4.17 s**
   warm versus **12.66–17.11 s** cold. Cold-versus-warm is not resolvable above
   foreign device load in this campaign, so the receipt reports both and leans
   on neither.
2. **The native endpoint is thread-dependent** — 5 thread counts, 5
   bitwise-distinct endpoints, 8.01% objective spread (§2). This is the
   recorded `collector-env-threading-fork` phenomenon, now confirmed at example
   scale and directly load-bearing for the band design.
3. **The GPU route showed no stochastic draw here.** The example's docstring
   pre-registers a 3-attempt budget because the latch is a draw (2/3 on an
   A100). On this RTX 5090, 8 of 8 legs latched on the first attempt with a
   bitwise identical endpoint. The retry budget was never spent on GPU — and
   was fully spent, unsuccessfully, on CPU.
4. **Two GPU legs have anomalous non-engine cost, not one.** Across all eight
   full-budget GPU legs the wall-minus-engine gap is
   `[20.091, 20.888, 21.257, 23.173, 24.890, 26.450, 56.560, 161.413]` s
   (ascending sorted, `artifacts/derived.json → wall_minus_engine_gaps`). Six
   form a 20.09–26.45 s cluster; **two sit outside it**:
   * `ab-r1-projected-gpuwarm` at **161.413 s** (wall 330.143 s, engine
     168.730 s). It ran immediately after the shipped-default native leg was
     SIGKILLed, while the 1-minute load was still decaying from 76.3. It
     supplies **both** §3's warm maximum **and** the engine minimum behind the
     most-charitable 3.266 reading.
   * `probe-projected-warm` at **56.560 s** — and this is the leg that most
     deserves naming rather than folding into a range, because it is *also*
     one of the four legs with **no sampled contention record** (§7) *and*
     **the GPU warm minimum, 235.246 s**. An earlier revision of this receipt
     quoted the other legs' span as "20.091–26.450 s", which was produced by a
     harness constant that silently excluded exactly this leg; the true span
     over the other seven is **20.091–56.560 s**. The exclusion is fixed in
     `derived.py`, which now computes the gaps over every such leg.

   No headline moves either way: dropping either leg leaves the verdict a
   bounded negative at every boundary, and both are retained because excluding
   a leg for being inconvenient at one end while it is also the most
   GPU-favourable at the other would be selection, not measurement.

---

## 9. Device-assignment change

`docs/jax_example_device_assignment.md`, row
`projected-route-single-stage-boozer-vacuum-optimization`:
`unmeasured / unmeasured` → **`gpu` / `measured-diagnostic`**, with a dated log
entry appended in the same edit, per that document's amendment procedure.

The placement follows that document's own stated semantics — *"if I want this
example to finish fastest, where do I launch it?"* — and §5 answers it without
ambiguity. The row's mechanism cell also carries §3's bar, so no reader can
take the row as a claim that this script is the fast way to this physics.

**Drift gate outcome — measured, before and after the edit.**

| | result |
|---|---|
| before the edit (`dd5e3113b`, clean tree) | `11 passed` |
| after the edit | `1 failed, 10 passed` — the only failure is `test_gpu_rows_cite_a_tracked_receipt_file` |

That single failure is **expected-pending-commit**, and it is the gate working
exactly as designed: a `gpu` row must cite a git-*tracked* file under
`docs/receipts/`, and this receipt is untracked until the orchestrator commits
it. Nothing else regressed — in particular `test_every_cited_in_repo_path_exists`
passes, which confirms the row's citation is parsed as an in-repo path and that
the path resolves; only `git ls-files` membership is missing. Committing this
file in the same commit as the row edit clears it, which is the procedure the
document's own amendment section prescribes.

Also updated in the same edit, per that procedure: the summary counts
(`1 gpu / 25 cpu / 13 unmeasured` → `2 gpu / 25 cpu / 12 unmeasured`; the
`native-*` mirror sub-count is unchanged because this example is not a mirror),
the in-repo evidence list, and the host-local artifact list.

---

## 10. Artifact index

Campaign root: `~/simsopt-campaigns/projected-route-example-promotion-20260816/`
(41 leg JSONs, 37 of them carrying a sampled contention record; 15 harness
files; 27 artifact files). **[host-local — not in this repository, not
reviewable from a clone.]**

| Path | Contents |
|---|---|
| `artifacts/receipt_numbers.json` | **169** numerals, each bound to a source file, a JSON pointer and that file's sha256; regenerating re-reads every source, so a stale binding fails loudly. **168 of the 169 carry a digest of the file they were read from; one cannot** — `audit_binding_count` binds into this same file, so no value it could store would be a digest of the version containing it, and its recorded digest is necessarily one generation stale |
| `artifacts/receipt_numeral_audit.json` | the reverse direction, **v2**: every distinct numeric token in this receipt — integers included — classified as bound, arithmetic-on-bound, explicitly cited from a named source, or document structure such as a section number or date part, with `unmatched_after_audit` **empty**. The exact per-class counts are in the artifact rather than quoted here, because the audit reads this receipt and quoting its own totals back into the text makes them a moving fixed point. **Two tokenizer defects in v1 are fixed and disclosed in the artifact**: v1 required a decimal point or an exponent, so every integer was exempt by construction, and a trailing `x` made ratios **invisible** — `5.07x` yielded no token at all, because the trailing-identifier guard rejected the match outright rather than shortening it, so the headline ratios were never presented to the audit in any form. The v1 claim "all 114 distinct numeric tokens" was true only of the tokens v1 could see |
| `artifacts/campaign_inventory.json` | the campaign's self-counts — leg/harness/artifact file counts, which legs lack a sampled contention record and which published statistics rest on them, the `run_leg.py` digest census, and the band-file-versus-probe mtime gap |
| `artifacts/contention_bias.json` | the in-campaign test of which lane contention favours (§7): per-lane wall spread, load correlation, and the per-round ratio-versus-load correlations |
| `artifacts/scoreboard.json` | per-configuration walls (ascending sorted), best-native selection, contention-matched per-round ratios, recorded failures |
| `artifacts/derived.json` | boundary ratio family, iteration accounting, matched-budget device comparison, threading fork, GPU endpoint reproducibility, matched-endpoint ratio |
| `artifacts/equivalence_final.json`, `artifacts/equivalence_second_endpoint.json` | the seven gates on two independent endpoint pairs, with the four native-evaluator measurements each |
| `artifacts/time_to_band.json` | the native budget ladder of §4 |
| `artifacts/terminated_legs.json` | the shipped-default leg's lower bound and why it was terminated |
| `artifacts/native_endpoint_reproducibility.json`, `artifacts/gpu_endpoint_reproducibility.json` | endpoint digests per leg |
| `artifacts/gpu_start.txt`, `artifacts/ps_start.txt` | the exact pid allowlist captured at campaign start |
| `legs/*.json` | every leg, each stamping commit, branch, tree cleanliness, python/jax/numpy/scipy versions, the executed example's sha256, the full environment override set, and before/after/sampled box state |

Harness — **15 files, all 15 enumerated**, sha256 read from the live files after
every other edit in this revision had landed (the list is refreshed last on
purpose: editing `numeral_audit.py` changes its own digest, so any list written
before it is stale by construction):

| File | sha256 | Role |
|---|---|---|
| `analyze.py` | `cedcc362…` | scoreboard, failures, contention-matched rounds |
| `bands.py` | `57f1d782…` | the pre-registered band registry (§2) |
| `boxstate.py` | `a2530702…` | pid-allowlist box state |
| `campaign_inventory.py` | `9be243c7…` | the campaign's self-counts |
| `contention_bias.py` | `ab0239bc…` | the in-campaign bias test (§7) |
| `derived.py` | `36ef0e88…` | boundary ratios, iteration accounting, gaps |
| `equivalence.py` | `cbf456ab…` | the seven-gate battery (§6) |
| `native_evaluator.py` | `079b07fa…` | the native evaluator replica |
| `numeral_audit.py` | `b621e358…` | this receipt's numeral audit |
| `phase2.sh` | `fdcd5f1b…` | thread optimum, cold lane, CPU backend |
| `phase3.sh` | `b8808f42…` | time-to-band and equivalence |
| `receipt_numbers.py` | `71925821…` | the numeral bindings |
| `run_leg.py` | `0d003d8a…` | one supervised leg (latest of three digests) |
| `sweep.py` | `988bd2a4…` | the interleaved A/B driver |
| `time_to_band.py` | `0d09452b…` | the native budget ladder (§4) |

An earlier revision of this list said "15 files" while enumerating 13, omitted
`campaign_inventory.py` and `contention_bias.py` entirely, and published two
digests that no longer matched their files — including, self-defeatingly, the
pre-fix digest of `numeral_audit.py` inside the paragraph disclosing that
tool's own defects.

`run_leg.py` was edited **twice** during the campaign — first to add the
whole-leg contention sampler, then to add the `projected_cpu` lane — so the 41
legs stamp **three** distinct `run_leg.py` digests, not one and not two:
`2e138ece…` (4 legs: the two plumbing legs and the two first probes, which are
exactly the four with no sampled contention record, §7), `592ad3dc…` (6 legs:
interleaved round 1), `0d003d8a…` (31 legs: everything after). Each leg carries
its own digest, which is what the per-leg stamp is for; the counts are in
`artifacts/campaign_inventory.json`.

---

## 11. Reproduction

```
cd ~/simsopt-campaigns/projected-route-example-promotion-20260816
# one interleaved A/B round (native OMP=8, OMP=16, GPU warm)
python harness/sweep.py --rounds 1 --plan "native:8,native:16,projected:-" --prefix repro
# endpoint equivalence through the native evaluator
env -i PATH=/usr/bin:/bin HOME=$HOME MPI4PY_RC_INITIALIZE=false \
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=8 \
    .../.venv-qn-gpu/bin/python harness/equivalence.py \
      --native-leg legs/<native>.json --projected-leg legs/<gpu>.json \
      --out artifacts/<out>.json
# scoreboard, derived quantities, numeral bindings, and the receipt numeral audit
python harness/analyze.py && python harness/derived.py \
  && python harness/campaign_inventory.py && python harness/contention_bias.py \
  && python harness/receipt_numbers.py && python harness/numeral_audit.py
```

`receipt_numbers.py` exits nonzero if any binding cannot be resolved;
`numeral_audit.py` exits nonzero if any numeral in this receipt — **integers and
ratio-suffixed numbers included** — is neither bound, nor arithmetic the receipt
states on bound values, nor an explicitly listed citation, nor a listed
document-structure token. Both exit 0 as of this writing. Run
`numeral_audit.py` last: it reads this receipt, and this receipt quotes its
counts, so the pair converges in one extra pass.

Drift gate, after the device-assignment edit:

```
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPI4PY_RC_INITIALIZE=false \
  .venv-qn-gpu/bin/python -m pytest tests/test_jax_example_device_assignment.py -q
```

---

## 12. Next action

**Re-time the 287.30 s native bar under a stated `OMP_NUM_THREADS`** (§8). It is
the one number this campaign found that no artifact in the repository can
currently reproduce, it is the denominator of a certified 2.304x claim, and the
campaign identified a mechanism — the unpinned 64-thread OpenMP collapse, worth
more than 28.78x on this box — that could account for the entire 4.46–5.76x gap
in either direction. Nothing else in the projected-route ledger should be
re-litigated before that measurement exists.
