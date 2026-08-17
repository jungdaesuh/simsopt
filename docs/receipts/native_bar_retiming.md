# Native bar retiming — diagnostic receipt

> **DIAGNOSTIC, NOT CERTIFYING.** One measurement campaign, one workstation,
> **2026-08-17 UTC**. Nothing here is sealed and no digest here is frozen into a
> gate. This receipt re-times the **denominator** of the certified projected-route
> claim at today's HEAD on this box. It does not re-run the numerator, it does
> not edit a sealed record, and it does not retract one: the 2026-08-13
> certificate stands as history and its arithmetic is reproduced here exactly.
> What changes is what the denominator *means*. Stated in the form the finding
> should be quoted in:
>
> **The 287.30421751597896 s bar was produced with no thread variable set, on a
> 48-CPU affinity mask, on a box whose quiet gate had been
> bypassed. Under `OMP_NUM_THREADS=8` the same interior region of
> the same lane takes 47.998 s — the bar is inflated
> 5.986x — under which the certified 2.304x becomes
> 0.385x, an inversion with native faster by
> 2.598x at the certified boundary.**

| | |
|---|---|
| Question | Does the sealed native bar `warm_synchronized_solve_max_s = 287.30421751597896` survive re-timing **at its own boundary, in its own lane**, under a pinned `OMP_NUM_THREADS`? |
| Verdict | **NO — the bar is inflated 5.986x.** Its own lane, its own interior boundary, the same problem bytes, at the best pinned thread count (`OMP_NUM_THREADS=8`, n=4) takes a median **47.998 s** (range 47.533–48.183 s). Every fast rung of the ladder — {4, 8, 16} threads — sits 5.317–5.986x under the bar. **The certified 2.304x INVERTS to 0.385x**, and so does every other boundary in the certificate's published ratio family. |
| Mechanism | **Recovered from the bar campaign's own controller logs, not inferred.** The bar's collector was launched under `taskset --cpu-list 0-23,32-55` — 48 of the box's 64 CPUs — with **no thread variable set anywhere**, so OpenMP sized itself from the affinity mask; the launcher's CPU-quiet gate was **explicitly bypassed** at that launch; and a foreign workload was being actively pinned to the complementary 16 cores throughout the native lane's window. |
| Repository | branch `pr/jax-port-squashed` @ `11f63f9fcf0b1d7fe6e0ed952b8989c025c65ca2` — one commit across every leg. **Not "clean at every leg", because that would be false:** 10 legs ran against a clean tree and 11 ran after this campaign's own draft receipt landed in it. The only path ever dirty is `?? docs/receipts/native_bar_retiming.md`; `M docs/single_stage_jax_gpu_projected_route_certification_plan.md`, which no lane imports. Stamped per leg. |
| Interpreter | CPython 3.11.15, jax 0.10.0, numpy 2.4.6, scipy 1.17.1, `.venv-qn-gpu` |
| Box | RTX 5090 + 64-core host, **shared with a foreign workload for the whole campaign** (§8) |
| Artifacts | `~/simsopt-campaigns/native-bar-retiming-20260817/` — 21 leg JSONs (21 carrying a sampled contention record; all-legs = True), 21 trajectory JSONLs, 16 harness files, 11 derived artifacts |

---

## 1. What the bar is, exactly

The bar is not a process wall and never was. It is the
`wall_seconds_from_start` field of the **final row** of one native lane's
`trajectory.jsonl`, written by `OptimizationTrajectoryRecorder` in
`src/simsopt/optimization_trajectory.py`, whose `perf_counter` starts at
recorder construction (`:137`) and is read on every recorded row (`:164`,
`:180`). In the executed lane that recorder is constructed by
`OptimizationMeasurementWindow` inside
`examples/jax/parity/cases/native_boozerqa.py::_native`, immediately before the
lane's required initial value+gradient evaluation and the scipy BFGS call. So
the timed region is

> recorder construction → initial value+gradient → scipy BFGS → 1000th callback

and it **excludes** process spawn, interpreter and library imports, the NCSX
build, the initial Boozer Newton solve, the endpoint evaluation, and receipt
publication.

That value, 287.30421751597896, is the sealed authority in
`docs/single_stage_jax_gpu_sqp_primal_dual_phase0_budget.json` and
`docs/single_stage_jax_gpu_coupled_fullspace_filter_trust_region_phase0_budget.json`
(`gates.warm_synchronized_solve_max_s`), is duplicated in
`src/simsopt_jax/solve/fullspace.py`, and is pinned by
`tests/jax/solve/test_fullspace_route_contract.py`. Its producing run is the
`native-cpu` lane of the `single-stage-speed-20260804` campaign, whose
`lane_result.json` and `trajectory.jsonl` re-hash here to the digests published
in `docs/single_stage_speed_campaign_results.md`. That run self-describes as
`authoritative: false`, `repository_dirty: true`,
`normalized_status: budget_exhausted`, `nit = 1000`,
`nfev = 1274`, final objective 4.4822246533126125e-08 — and
`lane_environment_policy: {}`, which is why no thread setting was recoverable
from the receipt itself.

---

## 2. The retimed lane is the bar's own lane, and it solves the bar's problem

Not asserted — bound, four ways.

**(a) The child entrypoint is the same one.** Every leg runs
`python -m examples.jax.parity.child --case
native-single-stage-boozer-vacuum-optimization --lane native-cpu --scale
native_default --trajectory-path … --optimization-timing-path …`. That is the
same case, lane and scale the sealed run executed, driven through the same
recorder, so the interior number this campaign reads is produced by the same
mechanism that produced the bar rather than by a re-implementation of it.

**(b) The problem bytes are identical.** The input bundle built at today's HEAD
reproduces the sealed run's `input_fingerprint`, its
`configuration_fingerprint`, and the payload digests of its `construction:coil_dofs`
and `construction:surface_dofs` arrays — all equal, in
`artifacts/equivalence.json → problem_identity`.

**(c) Every leg reproduces all three sealed fingerprints and the sealed
termination shape.** For 20 completed legs the
`input_fingerprint`, `configuration_fingerprint` **and**
`effective_construction_fingerprint` equal the sealed run's, the driver and the
six completed workflow stages equal the sealed run's, and every leg ends
`budget_exhausted` at `nit = 1000` with a 1000-row trajectory — the
sealed run's own termination. The construction fingerprint is the strong one: it
digests the seed together with the values reconstructed from the *instantiated*
case, so it binds the built problem, not the file on disk.

**(d) Cross-campaign, the endpoints match bitwise.** 17 of
17 legs run at a thread count the 2026-08-16 promotion
campaign **also** measured published a final objective that is **the exact
float64** that campaign published for the *shipped example*
`examples/3_Advanced/single_stage_boozer_vacuum_optimization.py` at the same
thread count. Those were different processes, a different script and a different
campaign. Bitwise agreement at matched threads is what makes this campaign's
interior numbers and that campaign's process walls two boundaries on one
computation rather than two computations. Thread counts this campaign swept that
the promotion campaign did not are **out of scope for this binding, not
mismatches**: 48.

---

## 3. Executed-source delta, stated as a delta

The two runs are not byte-identical and this receipt does not pretend they are.
The sealed run published 2295 executed sources with
digests; here is exactly what moved.

**The compiled kernel did not move at all.** The sealed lane's
`simsoptpp` extension digests to `41b2ca791a720f325ffa9b382b31d29bade73f6516693805d41adc0de6f6ed4b`; the installed extension
re-hashes to the same value today (`byte_identical = True`).
This workload's time is spent in those C++ Biot-Savart and Boozer-residual
kernels, and they are literally the same binary the bar ran.

**Of the 72 tracked `src/simsopt/` files the sealed
run executed, every one was clean at the bar's commit** (each executed digest
equals that commit's blob content), so all of them can be diffed exactly.
64 are byte-identical at HEAD. 8 changed,
and all 8 diffs were read:

* `_core/json.py` — a serializer refactor confined to `as_dict`, which this lane
  never calls;
* `_core/optimizable.py`, `configs/zoo.py`, `geo/curve.py`, `geo/surface.py`,
  `field/sampling.py`, `field/tracing.py` — docstring and typo corrections only;
* `geo/boozersurface.py` — an exact-Newton **observation sink**, entirely inside
  `if observing:` branches. With no sink installed the Newton body executes the
  same two `np.linalg.solve` calls the bar's bytes executed. At HEAD the sink has
  exactly two callers, a benchmark and a test, and neither is imported by
  `examples.jax.parity.child`.

**What is not recoverable, said plainly.** The bar's tree was dirty. Its
`src/simsopt/optimization_trajectory.py` was *untracked* at that commit, and its
`examples/jax/parity/{child,cases/native_boozerqa,cases/native_single_stage_boozer_vacuum}.py`
executed digests differ from that commit's blobs — the trajectory instrumentation
lived in the uncommitted diff. Those exact bytes cannot be reconstructed from the
repository. They are the measurement shell, not the computation: the physics
files were clean and are accounted for above. The residual this leaves is
one-evaluation-sized — whether the bar's shell put the initial value+gradient
inside the window, as HEAD's does. Bounded from the artifacts themselves, that
evaluation is at most the bar's own first trajectory row, 0.507733 s
out of 287.30421751597896 s.

---

## 4. The OMP ladder

### Ladder — interleaved rounds, interior boundary (the bar's own)

| OMP_NUM_THREADS | n | interior median (s) | interior min (s) | interior max (s) | wall median (s) | wall − interior median (s) | sealed bar ÷ interior median |
|---|---|---|---|---|---|---|---|
| 4 | 4 | 51.411 | 50.615 | 53.050 | 53.240 | 1.826 | 5.588x |
| 8 | 4 | 47.998 | 47.533 | 48.183 | 49.798 | 1.800 | 5.986x |
| 16 | 4 | 54.031 | 53.507 | 54.409 | 55.860 | 1.829 | 5.317x |
| 32 | 4 | 234.846 | 233.734 | 235.273 | 236.987 | 2.141 | 1.223x |

### Ladder — endpoint and work identity

| OMP_NUM_THREADS | n | distinct final objectives | distinct nfev | 1-min load mean, range over legs | foreign CPU %, mean per leg, range |
|---|---|---|---|---|---|
| 4 | 4 | 4.5285091875e-08 | 1256 | 12.39–49.17 | 425.0–919.4 |
| 8 | 4 | 4.3058760577e-08 | 1239 | 15.21–46.90 | 419.0–868.2 |
| 16 | 4 | 4.5246349670e-08 | 1274 | 21.89–38.76 | 413.2–824.4 |
| 32 | 4 | 4.4423838366e-08 | 1250 | 54.07–56.62 | 443.5–730.0 |

### Non-interleaved legs, published in full

| tag | role | OMP | interior (s) | wall (s) | final objective | 1-min load mean |
|---|---|---|---|---|---|---|
| `p2-omp48` | phase-2 probe of the sealed bar's own conditions | 48 | 551.710 | 554.409 | 4.4822246533e-08 | 82.93 |
| `p3-barconditions-b` | phase-2 probe of the sealed bar's own conditions | unset | 1513.310 | 1517.650 | 4.4822246533e-08 | 59.48 |
| `p3-barconditions` | phase-2 probe of the sealed bar's own conditions | unset | 1501.714 | 1505.489 | 4.4822246533e-08 | 57.44 |
| `plumb-omp8` | pre-round plumbing leg | 8 | 49.746 | 51.670 | 4.3058760577e-08 | 10.22 |

### Bar-conditions probes

| tag | conditions | state | interior (s) | wall (s) | 1-min load mean |
|---|---|---|---|---|---|
| `p2-omp48` | OMP_NUM_THREADS=48, no affinity mask | completed | 551.710 (1.920x **over** the bar) | 554.409 | 82.93 |
| `p2-ompunset` | threads unset, no affinity mask (the shipped default here) | **terminated at its 2400 s timeout** | > 2388.636 (lower bound, 779 of 1000 iterations) | > 2400.119 (lower bound) | 74.64 |
| `p3-barconditions` | the bar's own conditions: taskset 0-23,32-55, threads unset | completed | 1501.714 (5.227x **over** the bar) | 1505.489 | 57.44 |
| `p3-barconditions-b` | the same, repeated | completed | 1513.310 (5.267x **over** the bar) | 1517.650 | 59.48 |

### The certified ratio family, recomputed against the retimed bar

| GPU boundary (sealed, not re-run) | GPU (s) | ratio vs sealed 287.30421751597896 s bar | ratio vs retimed bar | native faster by |
|---|---|---|---|---|
| warm engine compile+solve (certified) | 124.707842 | 2.304x | 0.385x | 2.598x |
| warm attempt wall (child timer) | 156.856340 | 1.832x | 0.306x | 3.268x |
| warm supervised wall (Popen-exit) | 158.741789 | 1.810x | 0.302x | 3.307x |
| cold-lane engine compile+solve | 170.693673 | 1.683x | 0.281x | 3.556x |
| cold-lane attempt wall (child timer) | 226.003532 | 1.271x | 0.212x | 4.709x |
| cold-lane supervised wall (Popen-exit) | 228.083841 | 1.260x | 0.210x | 4.752x |

**Rounds are interleaved, and the ladder statistic uses only interleaved legs.**
Each round runs every configuration once, with the order rotated per round.
This box carries a foreign workload whose duty cycle drifts on a minutes
timescale and the repo's own receipts record batched native drift up to 53%;
running one configuration's repeats back to back would alias that drift onto the
configuration axis. The pre-round plumbing leg and the phase-2/3 probes are
complete measurements too, but they are not interleaved, so they are published
separately in full rather than folded into a configuration's median.

**The interior boundary and the process wall are nearly the same thing for this
lane.** Wall minus interior is ~2 s on every leg, which is what the boundary
analysis predicted and which independently corroborates the 2026-08-16 audit's
bound of ≤ 289.90 s on the bar child's whole process wall (a gap of 2.60 s,
0.91%).

**The slowdown is uniform across the run, not a stall.** Against
`r4-omp8`, the median leg of the best pinned rung, the sealed bar's
per-iteration cost is 5.658x higher (0.223386 s
versus 0.039483 s), and quarter by quarter of the 1000
iterations the ratio is 5.637, 6.170, 5.851, 6.295 — a band of
5.637–6.295x with no quarter carrying the
excess. A one-off stall would not look like this; a systematically more expensive
evaluation would.

---

## 5. Where the certified 2.304x lands

**It inverts.** The certificate's own ratio family, recomputed against the
retimed bar, is the fourth table in §4. At the certified engine boundary the GPU
side is 124.707842 s; against the sealed bar that is
2.304x, and against the retimed bar it is
0.385x — **native is faster by
2.598x**. At the certificate's strictest published
boundary the 1.260x becomes 0.210x, native
faster by 4.752x. Every row of the family inverts;
none is close to unity.

The comparison is boundary-symmetric in exactly the sense §12.14 of the
certification plan claims for itself. That section defends quoting the engine
boundary on the grounds that "the 287.304218 s bar is *itself* an interior
time-to-quality figure … so both sides exclude their own process bootstrap." That
reasoning is correct and it is preserved here: the retimed number is the same
interior figure, produced by the same recorder, in the same lane. The
denominator changed; the boundary did not.

The arithmetic of the sealed certificate is reproduced, not disputed:
`analysis.json → adjudication.gpu_ratio_family_against_retimed_bar` recomputes
2.304x, 1.832x, 1.810x, 1.683x, 1.271x and 1.260x against the
sealed bar from the certificate's own seconds. The certificate is sound against
its bar. Its bar is not a properly-threaded native reference.

---

## 6. The bar's launch conditions, recovered from primary artifacts

The promotion campaign could only state the unpinned-thread mechanism as a
hypothesis, because the bar's receipt records `lane_environment_policy: {}`. It
does not have to stay a hypothesis. The campaign that produced the bar left a
controller directory beside it —
`~/simsopt-campaigns/single-stage-speed-20260804-controller-r1/` — holding the
launcher, its watcher log and an affinity-guard log. Read verbatim into
`artifacts/bar_run_conditions.json`:

* **The launch line.** `2026-08-05T01:25:33-04:00 launching campaign; campaign_cpuset=0-23,32-55` — the collector ran under
  `taskset --cpu-list 0-23,32-55`, i.e. **48 of the box's
  64 CPUs**. The sealed native lane's `lane_result.json` was
  written at 2026-08-05T01:30:25-0400, inside that launch.
* **No thread variable, anywhere.** The launcher sets none. At the bar's commit,
  `benchmarks/run_jax_native_example_measurements.py`'s
  `build_measurement_environment(profile_id="native_cpu")` sets only `PYTHONPATH`
  and `MPI4PY_RC_INITIALIZE` on a scrubbed copy of the parent environment. So the
  child inherited `OMP_NUM_THREADS` unset and OpenMP sized itself from the
  affinity mask — 48 threads.
* **The quiet gate was bypassed.** The watcher logged
  `CPU quiet gate bypassed by user authorization` at that same launch. The box was
  not verified quiet.
* **A foreign workload held the rest of the box.** The affinity guard was pinning
  a `coupled_matrix_2026-08-03` workload to the complementary
  `24-31,56-63` — 16 cores — and logged
  21 corrections during the native lane's window,
  including a burst of fresh pids seconds after launch.

This is the same phenomenon the repo already records as
`collector-env-threading-fork`, now with its magnitude priced at the flagship
denominator: **48 unpinned threads on a contended box cost
5.986x against 8 pinned threads.**

---

## 7. Does high-thread / unset-`OMP_NUM_THREADS` execution reproduce the bar's magnitude?

**Yes, and only there.** Computed over every configuration this campaign
completed (`analysis.json → bar_magnitude_bracket`), the bar is bracketed by

* `32` at 234.846 s — the bar is
  1.223x above it, and
* `p2-omp48` at 551.710 s — 1.920x
  above the bar,

so the 287.30421751597896 s magnitude is reproduced within a factor of two on the
high-thread side of the ladder (True) and **is not approached
at all** on the pinned side, where every rung at {4, 8, 16} threads sits
5.317–5.986x below it. On this box
the cliff between the two regimes lies between 16 and
32 threads.

Four probes stand behind that, none of them a guess: two sweep the high end of
the thread axis on the open box, and two reconstruct the bar's own `taskset`
mask with threads unset. They are in §4's **Bar-conditions probes** table. A
probe that hit its timeout is recorded there as a **lower bound** — the last
trajectory row it managed to write — never as a measurement and never folded
into a median.

**The reconstruction reproduces the bar's trajectory exactly and its wall not
at all, and the residual is unexplained.** The bar's mask (`0-23,32-55`) with
threads unset sizes OpenMP to 48; both `p3-barconditions`
legs re-run exactly that and land on the sealed endpoint **bit for bit** —
objective `4.4822246533126125e-08`, `nfev = 1274`, the same final trajectory
row the sealed run wrote — while taking 1501.714 / 1513.310 s where the sealed
run took 287.30421751597896 s. Both legs ran on a **foreign-quiet box** (foreign
CPU 0.0% across all 729 / 735 contention samples; the foreign workload was
absent during their window), so foreign contention cannot explain the
residual ~5.2x; candidate causes this campaign does not adjudicate include
frequency/boost and thermal differences between the two nights and SMT
scheduling on the masked sibling pairs. What the probes do establish is
**regime membership, twice over**: the sealed endpoint objective is produced
only by the 48-thread-class runs (`p2-omp48` and both `p3` legs) and by **no
pinned rung at 4-32 threads** — so the sealed bar's trajectory is a
48-thread-class trajectory by bitwise fingerprint — and every 48-thread-class
probe lands far above the properly-threaded 47.5-54.4 s, on the same side of
the cliff as the bar. Read the probes as fixing the bar's *regime*, not as
replicating its number: the exact 287.30421751597896 s belongs to its night.

---

## 8. Contention, honestly

**The box was never quiet.** A foreign workload — `scripts/run_trial.sh`, rooted
at `/home/jungdaesuh/code/fusion/fusion_equilibrium_challenge`, relaunching
itself in a loop with fresh pids — held the GPU and a large share of the host for
the entire campaign. Measured per leg over the whole leg (2 s sampling): 1-minute
load mean 10.22–82.93 across legs, and the foreign
workload's summed `pcpu` averaged 0.0–1012.9% per
leg (median across legs 529.6%). Every one of the
21 legs carries a sampled contention record — all-legs coverage
is True, 21 of
21 — plus an exact-pid box-state pair before and after.

`harness/boxstate.py` classifies GPU processes by an **exact pid allowlist
captured at this campaign's start plus self-pid exclusion**, never by substring
match on a process name — the disclosed 2026-08-16 defect. The allowlist was
re-captured today rather than copied from the previous campaign, because pids do
not survive a session and reusing stale literals would have silently reclassified
today's desktop processes as contenders. The foreign workload present at start
was deliberately **not** baselined.

**Which way this biases the verdict, and it is the safe way.** Contention on
this campaign's side can only make the retimed lane *slower*. A slower retimed
lane means a **smaller** inflation factor and a ratio-vs-retimed-bar **closer to
unity** — so both headline numbers understate what a quiet box would show. The
5.986x and the 0.385x are floors on the effect,
not centre estimates.

That is not a claim that the comparison is fair in the other direction. §6 shows
the sealed bar also ran contended, and this campaign does not correct for that —
it discloses it. Nor does it license a symmetric argument about the GPU
numerator, which was measured on a third day under conditions neither run
recorded in a comparable form (§9).

---

## 9. What this does NOT settle

* **The GPU numerator was not re-run.** 124.707842 s is quoted verbatim
  from the sealed certificate. The contention conditions of that sealed run and
  of today differ and cannot be reconciled after the fact. What can be bounded:
  the 2026-08-16 campaign measured this GPU lane's wall dispersion at 1.489x
  max/min over 7 legs, which is nowhere near the
  2.598x it would have to cover. What cannot: a
  contention-matched head-to-head at today's HEAD, which this campaign did not
  run and which is the honest next experiment.
* **The bar's exact executed bytes are partly unrecoverable** (§3). The physics
  path is accounted for; the measurement shell is not.
* **One box, one interpreter, one day.** Nothing here is a hardware-general or
  version-general statement.
* **This is not a retraction of the sealed certificate.** It is a measurement
  that the certificate's denominator is not a properly-threaded native
  reference. The certificate's internal arithmetic is reproduced exactly in §5.

---

## 10. Two report-only documentation defects

Neither is fixed here — this campaign is forbidden from editing benchmarks or
plan prose, and both are recorded for whoever owns those files.

1. `benchmarks/run_single_stage_projected_route_gpu_root.py:23` states *"Native's
   287.30 s bar excluded nothing"*. That is false: §1's boundary analysis, and
   the executed bytes it rests on, show the bar excludes spawn, imports, the NCSX
   build, the initial Newton solve and the endpoint evaluation. §12.14 of the
   certification plan states the correct version.
2. The certification plan's §1 provenance table describes the bar as *"native
   reference wall for the same workload"*, while §12.14 correctly calls it an
   interior time-to-quality figure. The two cells disagree about what the number
   is; §12.14 is the later and correct one.

---

## 11. Artifact index

Campaign root: `~/simsopt-campaigns/native-bar-retiming-20260817/`
(**[host-local — not in this repository]**).

* `legs/*.json` — 21 legs (20 completed,
  1 failed), each stamping commit/branch/dirty, interpreter and
  library versions, the input-bundle digest, the digests of the four executed
  harness/case/recorder files, an exact-pid box-state pair, and a whole-leg
  contention series. Across all legs there is
  1 distinct interpreter/library set,
  1 distinct input-bundle digest and
  1 distinct case-module digest set — so no leg ran
  a different build of anything than any other leg.
* `legs/*.work/trajectory.jsonl` — 21 raw trajectories; each
  leg's interior number is the last row of its own file.
* `artifacts/analysis.json` — the ladder, the sealed bar re-derived from its own
  trajectory, the slowdown profile, and the adjudication.
* `artifacts/equivalence.json` — problem identity, per-leg fingerprint identity,
  the cross-campaign bitwise endpoint binding, and the executed-source delta.
* `artifacts/source_triage.json` — the compiled-kernel digest comparison and the
  per-file `src/simsopt/` diff sizes.
* `artifacts/bar_run_conditions.json` — §6's recovered launch conditions.
* `artifacts/tables.md` — §4's tables, generated.
* `artifacts/inventory.json`, `artifacts/receipt_numbers.json`,
  `artifacts/receipt_numeral_audit.json` — inventory and the numeral bindings.
* `harness/*.py`, `harness/*.sh` — 16 files; digests in
  `inventory.json`.

**This receipt is generated, not transcribed.** `harness/receipt_template.md`
holds the prose; `harness/render_receipt.py` substitutes every numeral from a
derived artifact and pastes `artifacts/tables.md`; an unknown or leftover
placeholder is a hard error. The prose and the artifacts cannot drift apart
because there is only one copy of each number.

---

## 12. Reproduction

```sh
cd ~/simsopt-campaigns/native-bar-retiming-20260817
PY=/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/.venv-qn-gpu/bin/python

# rebuild the bundle and check it against the sealed fingerprints
$PY harness/make_bundle.py bundle/native_default

# one interleaved round of the ladder
bash harness/sweep.sh 1

# the bar's own launch conditions
bash harness/phase2.sh && bash harness/phase3.sh

# derive, bind, audit, render
$PY harness/analyze.py && $PY harness/equivalence.py
$PY harness/source_triage.py && $PY harness/bar_run_conditions.py
$PY harness/inventory.py && $PY harness/tables.py
$PY harness/receipt_numbers.py && $PY harness/render_receipt.py
$PY harness/numeral_audit.py
```

---

## 13. Next action

**One experiment, and it is not another native leg.** The denominator is now
measured at three boundaries and five thread counts; the numerator is a sealed
number from a different day. Run the projected-route GPU lane and the pinned
native lane **interleaved, in the same rounds, on the same box, at today's
HEAD**, and publish the contention-matched per-round ratio. That is the only
measurement that can turn §5's inversion from *arithmetic against a sealed
numerator* into *a head-to-head result*.

Two ledger items fall out and are recorded rather than acted on:
`gates.warm_synchronized_solve_max_s` should not be used as a comparison
denominator again without a stated thread setting; and every campaign harness in
this repo that spawns a native child should pin its thread variables explicitly,
because the collector at the bar's commit demonstrably did not.
