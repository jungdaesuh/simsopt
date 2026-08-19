# Genuine-675 flat single-stage fair-bar campaign — preregistration

Status: **FROZEN** at commit `7b6d69041` (2026-08-18); amended by dated
pre-evidence amendments A1 (`2f0381cde`), A2 (`2a832c7a7`), A2a
(`0d1ca0607`), A3 (`e07bdc7c4`), A3a (below). Frozen-body text that an
amendment supersedes carries an inline pointer and is retained verbatim
per the amendment rule. Author date: 2026-08-18.

## Mission and verdict space

Adjudicate the archived claim that the fp64 JAX GPU lane of the
`GENUINE_FULLSPACE_675` flat single-stage formulation is ~10× faster than the
native SIMSOPT/simsoptpp CPU lane, by re-measuring under the program law
minted by the projected-route inversion and applied by the finite-build
successor campaign: **no native denominator without a stated, swept, and
pinned OMP**, symmetric priming, interleaved pairs, preregistered timers, and
fail-closed verdicts.

Terminal verdicts (exactly one per budget rung, defined below):

- `WIN` — the GPU lane beats the fastest fully-qualifying native
  configuration by ≥ 1.10 median on the primary timer with **every pair
  > 1.00**.
- `CLOSED_BOUNDED_NEGATIVE` — the protocol completed and the ratio rule
  failed. Reported as-is; supersedes the archived 9.8× in either direction.
- `NOT_PRODUCED` — any conformance, work-matching, or endpoint gate failed.
  No ratio is quoted from a `NOT_PRODUCED` run.

The kill criteria and budget caps in this charter are final. Post-evidence
clause amendment is prohibited (clause-shopping); defects discovered
mid-campaign close the affected phase and are disclosed.

## What is being adjudicated (the archived claim, fully stated)

Artifact: `~/simsopt_mixed_artifacts/genuine675-fixed-budget-maxiter3-r3-1c23f6c5-20260721T124425Z/`
(host-local), commit `1c23f6c5f8964c74cc60f63d81b7f93f2db852f3`, clean tree,
2026-07-21. Its three timers, all N=1: optimizer wall 53.603 s vs 5.471 s
(**9.797×**, the circulated number), process wall 58.702 s vs 7.859 s
(**7.469×**, the driver's own declared `comparison_timing_basis`), launcher
wall 60.700 s vs 10.033 s (6.05×). Matched work is emergent and exact at
maxiter=3: 9/9 compact evaluations, 3/3 accepted steps, endpoint objectives
agree to 5.46e-14 relative.

Defects this charter must fix (all verified against the artifact and its
driver sources, 2026-08-18 recon):

1. Native OMP never stated, swept, or pinned: the child env whitelist
   (`validation_ladder_common.py`, `MINIMAL_SUBPROCESS_BASE_ENVIRONMENT_NAMES`)
   carries no threading variable, so libgomp defaulted to 64 threads on 32
   physical cores with an independent unpinned OpenBLAS
   (`NO_AFFINITY`, `MAX_THREADS=64`) nested in the same process.
2. Priming asymmetry: each GPU lane ran a full discarded primer execution
   (23.3 s process wall, excluded from timing); the native lane ran once,
   cold, first.
3. N=1, fixed lane order (native → fp64 → mixed), no interleave. Native
   spread across the four sibling maxiter=3 triads is 1.53× (53.6–82.2 s)
   for bitwise-identical work.
4. Timer ambiguity: three nested timers spanning 6.05×–9.80×; the circulated
   9.8× is not the driver's own adjudication basis.
5. The input bundle manifest declares
   `performance_eligible: false` — enforced fail-closed by
   `fixed_state_genuine_675_input_manifest.py` — and the triad's own
   `claim-qualification.txt` says "Diagnostic fixed-budget evidence only."
6. `adjudication.passed = false` (the mixed lane failed its clause); the
   cited numbers come from a failed triad.
7. No machine/OS/driver/thread provenance recorded; the OpenMP runtime
   (`libgomp`) resolves through inherited `LD_LIBRARY_PATH` to an
   out-of-artifact pixi env.
8. Budget knife-edge: GPU overheads dominate at maxiter=3 — first-eval sync
   3.15 s plus certificate cost 1.54 s = 4.69 s = **86%** of the GPU's
   5.471 s optimizer wall (certificates alone 28%). The two components
   differ in kind: the first-eval sync is genuinely fixed, while the
   certificate cost is **event-driven** (2 attempts / 2 full certifications
   against 3 accepted steps, 3 skipped) and its cadence at larger budgets
   is unmeasured. Steady-state per-eval ratio is ~61× (native ~5.7 s vs GPU
   0.094 s), so the whole-budget ratio moves strongly with maxiter.

Mitigating archived evidence, recorded for calibration honesty: the cited r3
used the fastest of the four archived native samples (ratio 9.8× vs up to
15.1×), and the sole thread-count contrast on record (maxiter=1000 native
under `taskset -c 33-63`, i.e. ~31 SMT-sibling half-cores: last-20-eval mean
6.224 s vs r3's unpinned-64 steady state ~5.72 s) moved native only ~9%.
Nothing below 31 threads was ever measured; the sweep decides.

## Formulation identity clause (the two 675s)

This campaign binds exclusively to **`GENUINE_FULLSPACE_675`**
(`schema simsopt.single_stage.genuine_fullspace_675.formulation.v1`,
`formulation_semantic_sha256
0fe8e9e7fbabe69fa0dae82f92ca947304eef7b0d1c6a0050650c07282b4067a`): 675
active outer DOFs = coil 11 + vessel 3 + surface 661, inner state exactly
`(iota, G)` via a closed-form reduced-QR solve on a 48960×2 system,
`solved_inner_state.surface_dof_count = 0`, fail-closed decomposition guard
in `single_stage_fullspace_675.py`. The claim-ineligible
`HISTORICAL_PADDED_675_LAYOUT` (optimizer width 675, **active width 11**) is
a different object and is excluded; no number from any padded-layout runner
may enter this campaign.

Adjudication of the stale plan text: the 2026-07-20 ruling in
`docs/mixed_online_biotsavart_matrix_free_single_stage_implementation_plan_2026-07-20.md`
("genuine 675-active … not an existing runnable lane") was discharged the
next day by commits `8c986f466` (certify genuine 675 comparison campaigns),
`996d01bac` (cross-evaluate genuine-675 endpoints), and `63a7d33dc` (certify
genuine-675 fixed-budget evidence), which implement and certify exactly the
separate formulation that ruling required. The doc was never updated; this
charter records the discharge rather than obeying the stale text.

## Substrate and lineage disclosure

> **[Superseded by Amendment 1]** — the instrument is pinned detached at
> `1c23f6c5`, not the `5fb968188` branch described below.

The formulation does **not** exist on `pr/jax-port-squashed` (this branch's
`single_stage_fullspace.py` is the different 716-DOF NEQ problem; merge-base
with the measurement lineage is `fc28d62f8`). The campaign therefore runs on
a dedicated branch `genuine675-fair-bar` cut from **`5fb968188`**
(2026-07-22, the newest **placeholder-free** commit on the measurement
lineage that contains all three genuine-675 certification commits;
`src/simsopt_jax/runtime/single_stage_fullspace_675.py` is **byte-identical**
between `1c23f6c5` and `5fb968188`, so the physics is unchanged while the
driver carries the post-measurement certification machinery), in a **fresh
worktree** (`../simsopt-genuine675-fairbar`; imports of the formulation, the
triad driver, and the native simsoptpp stack smoke-verified there in the
pinned runtime env, 2026-08-18). The lineage tip `a79ddd4ea` is
**disqualified as a substrate**: its "dual-tree SSOT" merge replaced 27
`src/` files (including `simsopt_jax/__init__.py`) with non-Python
`common:`-pointer placeholder blobs, so its committed tree does not import. The existing
`simopt-jax-clean-local` worktree is disqualified as a run site: a concurrent
session's uncommitted migration has deleted the genuine675 files from its
disk (1836 dirty entries). Every gate-consuming run must validate clean-tree
at the campaign branch's harness commit. Receipts and the tracked evidence
bundle land on `pr/jax-port-squashed` (the program's receipt home, same git
object store), citing the campaign-branch commits; this three-tree situation
is disclosed in the receipt.

Runtime: the July campaign's proven environment —
`~/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2150Z/runtime-env/bin/python`
(Python 3.11.15, numpy 2.4.6, scipy 1.17.1, prebuilt
`simsoptpp.cpython-311-x86_64-linux-gnu.so`) with
`PYTHONPATH=<worktree>:<worktree>/src`. The harness must record, per timed
leg, from **inside the child**: `lscpu` summary, `nvidia-smi` name/driver,
`sched_getaffinity` mask, `omp_get_max_threads()`,
`openblas_get_num_threads()`, `jax.__version__` + x64 state, and the
resolved `libgomp.so.1` path from `/proc/self/maps` (defect 7).

## Input-bundle eligibility clause

The frozen r3 input bundle
(`~/simsopt_mixed_artifacts/genuine675-r3-input-1c23f6c5-20260721-r1/`,
manifest sha256
`84febc05d195d84c0802205b2b4c85ea1fa38faa7ff856efca7c12d980647c0c`,
`manifest_semantic_sha256 8dc7149e…`, six member files individually
sha-pinned and re-verified 2026-08-18) is the problem-bytes SSOT: same coil,
vessel, surface, equilibrium, and runtime-spec bytes, same initial state.

Its manifest carries **four frozen contract literals**, each enforced
fail-closed by the old loader, and this clause engages all of them rather
than relabeling one:

- `performance_eligible: false` and
  `classification: "non_timed_non_performance_fixed_state_parity_input"` —
  the old campaign's statement that *it* would mint no timing claim from
  this bundle. This campaign is a different preregistration making a
  **relative two-lane** claim: both lanes receive byte-identical inputs, so
  any input-quality concern cancels in the ratio. The reclassification is
  dated and pre-evidence; the old contract itself is not modified.
- `provenance_classification:
  "create_only_diagnostic_derived_from_native_event_29"` — the starting
  candidate is a mid-trajectory iterate captured from a native run. For a
  relative timing comparison this is a valid (and lane-neutral) workload;
  what it **limits is generality**: this campaign's verdict speaks to
  optimization launched from this one native iterate, not to an ensemble of
  starting points. The receipt must carry this scope sentence.
- `vessel.historically_authenticated: false` — disclosed; irrelevant to a
  two-lane relative measurement over identical vessel bytes.

Mechanically, the campaign mints a new campaign-scope manifest (schema
`genuine-675-fair-bar-input.v1`) that binds the identical member files by
their existing sha256s and declares `performance_eligible: true`. Because
the old loader cannot read the new schema, the harness ships a **new
loader** whose contract is preregistered here: it re-verifies all six
member-file sha256s and the byte counts against the campaign manifest and
fails closed on any mismatch (`NOT_PRODUCED`), and it is covered by a test
(tampered-byte and missing-file cases) that must pass before Phase 1 runs.
The member-file hashes may not change for any reason.

## Frozen protocol

### Lanes and shared policy

Two timed lanes only: `native_cpp_cpu` (simsoptpp/scipy, analytic Jacobian
via `dJ(partials=True)`) and `jax_gpu_fp64`. The mixed lane is excluded (it
is not work-matched — 8 vs 9 evals in the archive — and is not this
campaign's question). Both lanes share the archived policy bytes:
L-BFGS-B, `maxcor=300`, `maxls=8`, `ftol=0`, `gtol=1e-3`, unbounded,
`jacobian_supplied=true`, rolling-anchor accepted-state policy
(`shared_policy sha fc3498929ecdcbcb…`); the harness passes
`--expected-policy-sha256`/`--expected-formulation-sha256` so drift fails
closed.

### Work-matching and endpoint gates (promoted to hard gates)

Matched work is emergent from shared fp64 arithmetic; this campaign
**verifies it per leg and fails closed** (defect 8 of the archive — it was
observed, never gated):

- per-budget: native and GPU legs must report identical
  `compact_candidate_evaluations` and `accepted_callback_count`, **and**
  the two lanes' accepted-objective sequences must agree elementwise to
  relative ≤ 1e-10 (identical counts alone do not prove identical
  trajectories; both lanes already emit accepted callbacks, so this
  converts "same trajectory" from inference to evidence);
- endpoint objective relative difference ≤ 1e-10 (archived margin 5.5e-14,
  four orders in hand) and endpoint gradient-∞ relative difference ≤ 1e-8;
- the GPU endpoint is additionally re-evaluated through the **native
  evaluator** (oracle), and the oracle's objective must satisfy the same
  1e-10 clause against the native lane's endpoint objective;
- per-lane `y_certificate.accepted` and `precision_execution.admissible`
  (the existing driver gates) must hold;
- `termination_reason == scipy_completed` in every timed leg.

Any failure → that pair is `NOT_PRODUCED`; three `NOT_PRODUCED` pairs abort
the phase, and the affected budget rung reports `NOT_PRODUCED` — no ratio
of any kind is quoted from it.

### Budgets (both preregistered; caps final)

- **B3**: `maxiter=3` — continuity rung. **The adjudication of the archived
  9.8×/7.47× claim is B3 and only B3** (a maxiter=50 result cannot
  supersede a maxiter=3 claim).
- **B50**: `maxiter=50` — headline rung, and a **separate new claim** about
  a different budget, never a supersession of the archived number. The
  rationale is stated per component: the GPU's 3.15 s first-eval sync is
  genuinely fixed and amortizes; its certificate cost is event-driven with
  unmeasured cadence at 50 iterations, which is why the divergence probe
  below also records B50 certificate count and time **before** the rung is
  timed. No other budget may be scanned. **No ratio from this campaign may
  be quoted anywhere without its `maxiter` attached.**

### B50 divergence probe and contingency (pre-evidence, required)

Matched work is emergent and evidenced only at maxiter=3. Before any timed
B50 leg, one **untimed** native/GPU pair runs at B50, recording per-iterate
accepted-objective sequences, eval/accept counts, and per-lane certificate
count/time. Outcomes:

- Sequences match to the work-matching clause through iterate 50 → B50
  proceeds as chartered.
- The lanes fork at some iterate k < 50 → the headline rung becomes
  **B_k\***, where k\* is the largest budget at which the probe's sequences
  match, frozen by a dated amendment **before any timed leg at that
  budget** — the work-matching gate stays hard, the budget choice stays
  pre-evidence, and the rung neither dies silently nor gets rescued
  post-hoc. B3 is unaffected either way.

### Timers (preregistered; no timer shopping)

- **Primary: `process_wall_seconds`** — the archived driver's own declared
  `comparison_timing_basis`. Verdict rule applies to this timer.
- Secondary (reported, no verdict): `optimizer_seconds` and steady-state
  per-eval time (mean of the last max(3, N−3) evaluations).
- Cache/compile policy: the campaign's claim scope is **warm
  persistent-cache repeated workloads** (as the finite-build receipt's).
  Both lanes receive an **identical discarded primer leg** per timed process
  (defect 2 fixed symmetrically): the GPU primer warms the persistent XLA
  cache exactly as archived; the native primer is the same full lane
  execution, discarded. One fresh-cache disclosure pair (no primers, both
  lanes cold) is run per budget and reported without entering the verdict;
  no cold-start claim is minted unless the cold pair itself satisfies the
  win rule.

### Phase 1 — conformance and baseline

One untimed pair at B3 verifying: child-observed env conformance (fp64 on
both lanes, pinned threading vars echoed from inside the child), machine
provenance capture, work-matching + endpoint + oracle gates green, policy
and formulation shas match, input member hashes match. This pair also
reports the **native primed-vs-cold delta** (one cold and one
primer-preceded native leg, both untimed), so the symmetric-priming clause
is evidenced rather than formal. Failure blocks the campaign
(`NOT_PRODUCED`). The B50 divergence probe (above) runs immediately after
Phase 1 passes.

### Phase 2 — native OMP matrix (the fair bar)

> **[Superseded by Amendment 2]** — under the partition the sweep is
> `OMP ∈ {1, 2, 4, 8, 16*}` on the reserved set, the unpinned disclosure
> leg is retired, and the archived anchor covers the high-thread regime.

Native lane only, at each budget: `OMP_NUM_THREADS ∈ {1, 2, 4, 8, 16, 32,
64}` with `OMP_PLACES=cores`, `OMP_PROC_BIND=close`,
`OPENBLAS_NUM_THREADS=1`, CPU affinity pinned to physical cores `0-31` for
configs ≤ 32 threads (never the `33-63` SMT-sibling half — archived trap),
full mask for 64; plus **one unpinned-default disclosure leg per budget**
reproducing the July condition (nothing set — the archived defect, measured
not assumed). N=3 reps per config, median decides; **per-config rep
dispersion (max/min) is published alongside the median** (archived native
dispersion on identical work is 1.534×, so a lucky-fast denominator is a
disclosed false-negative risk for the GPU, never a false-positive one).
The **denominator config** is the fastest configuration whose three reps
all pass the gates. The matrix is complete before any pair runs at that
budget; no post-pair re-sweeps.

Wall-time control (preregistered, direction pro-GPU so fixed here): the
full 7-config matrix runs at **B3** only. At the headline budget, the
matrix runs **only the configs whose B3 median is within 2× of the B3
best**; the excluded configs' B3 numbers are published in the receipt.
Excluding configs can only make the native denominator slower or equal,
so this narrowing cannot flatter the GPU's opponent.

### Phase 3 — five interleaved final pairs (per budget)

Five native/GPU pairs, alternating start lane pair-to-pair, serialized (one
timed process at a time), box-idle gate before every timed leg (fail-closed
on load — **superseded by Amendment 2's partition-integrity gate**), native at the frozen denominator config, GPU as archived
(`PRODUCTION_CUDA_CHILD_ENVIRONMENT` + per-campaign persistent cache dir).
Ratio per pair = native `process_wall_seconds` / GPU `process_wall_seconds`.

Verdict per budget: median of the five ratios ≥ 1.10 **and** every pair
> 1.00 → `WIN`; protocol complete but rule unmet → `CLOSED_BOUNDED_NEGATIVE`;
any gate failure per the work-matching clause → that pair `NOT_PRODUCED`
(three abort the rung as above). The receipt reports the pair-ratio
**min/median/max** per budget, never only "threshold satisfied" (at
archived ratios of 6–15× the 1.10 threshold is not the informative gate;
every-pair > 1.00 and the work-matching clauses are). Instrument cost
bound: **exactly one discarded primer + one timed leg per lane per timed
process**; ≤ 96 timed legs campaign-wide including the matrix (expected:
21 B3 matrix + narrowed headline matrix + 2 disclosure + 20 pair +
4 fresh-cache legs ≈ 55–70).

## Evidence and validation

Every run directory carries: per-leg JSON rows with sha256s bound in a
manifest, launch rows (`process_wall_seconds` from the launcher), the frozen
campaign input manifest, policy/formulation shas, git identity
(commit + dirty count — must be clean), and the child-observed provenance
block. Every row additionally embeds a **campaign-contract sha256** — the
hash over (charter bytes at the freeze commit, shared-policy sha,
formulation semantic sha, campaign input-manifest sha) — and `validate`
refuses any row whose contract sha differs from the frozen one, so archived
rows cannot be rescored against a foreign contract (the finite-build
campaign's per-row gate-sha lesson, imported). A `validate` entrypoint recomputes the phase verdict from the run
directory alone. The tracked evidence bundle under
`docs/receipts/evidence/genuine675_fair_bar/` (on `pr/jax-port-squashed`)
preserves run layout so the bundle self-validates, per the finite-build
precedent. The receipt states all three archived timers next to the new
measurement and supersedes the 9.8× explicitly, whichever direction the
verdict goes.

## Ops constraints

> **[Superseded by Amendment 2]** — the quiet-box wait is replaced by the
> reserved-CCD partition with foreign-compute confinement; timed legs are
> gated by partition integrity, not whole-box load.

Shared box: the harness's idle gate blocks timed legs while foreign
compute (e.g., the currently-running full-core native job) is live; runs
wait, never contend. `tail ---disable-inotify` for monitors; long chains via
nohup + detached liveness watcher; GPU serialized under the campaign lock as
archived.

Wall-time budget: **≤ 12 hours of box time** across all phases. Arithmetic
basis: B3 native legs ~60 s at high thread counts; a headline-budget native
leg at the archived 5.87 s/eval is ~300 s, and low-OMP configs may be
severalfold slower — which is exactly what the B3-based matrix narrowing
bounds. If the budget would be exceeded, the campaign stops at a phase
boundary and reports what completed; partial phases are `NOT_PRODUCED`.

## Amendment rule

Amendments are permitted only before the evidence they govern exists, must
be dated, must state their empirical basis, and append to this file. The
win/kill thresholds, budgets, timers, and the five-pair rule are not
amendable after the charter commit.

## Amendment 1 — instrument substrate (2026-08-18, pre-evidence)

Empirical basis: the first two Phase-1 smoke attempts (untimed; no timed
evidence existed) were rejected by the archived instrument's own
fail-closed input validator, which the recon had not surfaced:
`validate_frozen_genuine_675_input_bundle` requires (a) a **clean** git
checkout including untracked files, and (b) the checkout's HEAD to equal
the frozen bundle's `launch_source.commit_sha1` — exactly
`1c23f6c5f8964c74cc60f63d81b7f93f2db852f3`. The archived input bundle is
therefore consumable only by an instrument tree at the measurement commit
itself.

Amended substrate (strictly stronger continuity than the frozen section's):

- The instrument worktree (`../simsopt-genuine675-fairbar`) is pinned
  **detached at `1c23f6c5`** — the timed lane driver, runtime, and input
  bundle are **bit-identical to the archived r3 instrument**; the
  `5fb968188` pin and its campaign branch are retired (its +2080 driver
  lines are post-measurement certification machinery this campaign never
  needed).
- The harness (`benchmarks/genuine_675_fair_bar.py`) and oracle child
  (`benchmarks/genuine_675_fair_bar_oracle.py`) live on
  `pr/jax-port-squashed` — they cannot live in the instrument tree without
  dirtying it — and always run with
  `PYTHONPATH=<instrument>:<instrument>/src` plus a fail-closed guard that
  the imported `simsopt_jax` resolves from the instrument tree. Run
  artifacts are written outside the instrument tree
  (`~/simsopt_mixed_artifacts/genuine675_fair_bar/`).
- Every row records **two git identities**: the instrument commit (must
  equal `1c23f6c5`, clean) and the harness commit (`pr/jax-port-squashed`
  HEAD, clean for gate-consuming runs).

No timer, threshold, budget, gate, or eligibility clause changes.

## Amendment 2 — shared-box partition protocol (2026-08-19, pre-evidence)

Empirical basis: the box hosts a continuous multi-run foreign campaign
(~50 cores per run, runs launching back-to-back), so the frozen Ops
section's quiet-box condition is unavailable for the foreseeable window;
the operator directed the campaign to proceed without an empty box. No
timed evidence exists (Phase 1 and the divergence probe are untimed).
Timing the native denominator against a ~50-core foreign job would
reproduce the projected-route artifact class (a contended denominator can
only flatter the GPU), so proceeding requires structural isolation plus an
uncontaminated high-thread anchor — both defined here, both in the
anti-GPU direction.

**Partition.** The campaign reserves CPUs `{0–7, 32–39}` exclusively:
cores 0–7 with their SMT siblings — exactly one CCD (L3 instance 0) of the
9970X, giving the reserved set a private 32 MiB L3. All foreign compute
processes are confined (reversibly, `taskset -apc`) to the complement
`{8–31, 40–63}` (24 physical cores + siblings; the current foreign run's
~50 threads oversubscribe that set by ~1.04×, a ≈4% cost to the foreign
campaign). Before every timed leg the harness verifies, fail-closed:
(a) no process with recent CPU activity outside the campaign has an
affinity mask overlapping the reserved set; (b) reserved-set busy fraction
< 20% (measured from `/proc/stat` over a 3 s window); (c) GPU utilization
≤ 5% (unchanged). The runner re-confines any newly spawned foreign process
between legs; the harness gate is the authority (violation = the leg fails
closed). The whole-box load gate of Amendment-0 ops is superseded for this
campaign by (a)–(c).

**Native matrix under partition.** The sweep is restricted to what the
reserved set can measure fairly: `OMP ∈ {1, 2, 4, 8}` pinned to dedicated
physical cores `0–7`, plus `OMP=16` on `{0–7, 32–39}` (disclosed as
SMT-assisted — 16 threads on 8 physical cores). `OMP ∈ {32, 64}` is
unmeasurable under partition; its role is filled by the **archived
high-thread anchor** below. The per-budget unpinned-default disclosure leg
is dropped (meaningless inside a partition); the July condition is instead
represented by the archived samples themselves.

**Archived high-thread anchor (anti-GPU by construction).** The four
archived maxiter=3 triads (2026-07-20/21, unpinned 64-thread native, the
same instrument bytes) provide contention-free native samples: process
walls 58.702 / 77.046 / 82.039 / 87.310 s, optimizer walls 53.603–82.246 s,
fastest steady per-eval 5.867 s (r3: 52.807 s compact over 9 evals). The
anchor bars are: **B3 anchor = 58.702 s** (the fastest archived process
wall) and **B50 anchor = 66 × 5.867 = 387.2 s** (the fastest archived
per-eval times the measured B50 evaluation count, with zero overhead
added — an extrapolation constructed strictly in native's favor).
**[Derivation superseded by Amendment 2a** — the headline anchor is
pair-derived, not a fixed constant.**]**
Admitting these candidates can only lower the denominator bar, never
raise it, so no partition-induced slowdown of the live native legs can
inflate the ratio beyond what the archive licenses.

**Amended verdict rule (strictly stronger than the frozen rule).** A
budget rung is `WIN` only if BOTH hold: (1) the five interleaved
partition pairs satisfy the frozen rule (median ≥ 1.10, every pair
> 1.00) against the live swept denominator; **and** (2) the GPU lane's
median process wall beats the archived anchor bar by the same margin
(anchor / GPU-median ≥ 1.10). Failing (2) while passing (1) is
`CLOSED_BOUNDED_NEGATIVE` with the anchor cited. Both conditions are
reported with min/median/max.

**Symmetric residence.** GPU pair legs' host process is pinned to the
same reserved set, so both lanes of every pair live in the identical
partition. Primers, alternation, work-matching, endpoint, and oracle
gates are unchanged. Residual interference channel (shared DRAM
bandwidth with the foreign complement) is disclosed; its direction on
live legs is pro-GPU, which is exactly what condition (2) bounds.

No budget, timer, or pair-count changes. The instrument is unchanged.

## Amendment 2a — anchor derivation clarification (2026-08-19, pre-evidence)

Review of Amendment 2 (adversarial, pre-launch; no timed pairs exist)
surfaced three defects in the anchor paragraph, corrected here:

1. **Derivation.** The headline-budget anchor is **not** a fixed constant:
   it is computed as (the matched evaluation count measured by the
   campaign's own work-matched pairs at that budget) × (the fastest
   archived run's **sustained per-eval mean**, 52.807 s / 9 = 5.867 s).
   This works at any headline budget — including a B_k\* contingency rung —
   and the count it uses is validated by the work-matching gate on the
   very pairs it bars. The B3 anchor remains the archived process wall
   58.702 s (measured, no extrapolation). The "66 evaluations" figure in
   Amendment 2 was observed on **supplementary A100 GPU runs** (the GPU
   lane's count on different hardware) and is superseded by the
   pair-measured count; it survives only as corroboration.
2. **Direction, stated auditable rather than superlative.** "Constructed
   strictly in native's favor" is replaced by: the anchor uses the fastest
   archived run's sustained mean (within-run spread 5.591–6.353 s;
   a min-single-eval construction would give ~4.9% less) and excludes the
   archived 5.099 s of non-optimizer process overhead (~1.3% the other
   way); net ≈3.5% pro-GPU at most, immaterial at the expected ratio
   scale, and disclosed rather than asserted.
3. **Provenance, stated outright.** The headline anchor is an
   extrapolation never measured on any native lane at any thread count.
   The receipt must publish all four archived unpinned-64 process walls
   (58.702 / 77.046 / 82.039 / 87.310 s — spread 1.49×) beside the anchor,
   and if the selected live denominator is the SMT-assisted omp16 config,
   the receipt's selected-config line must carry that label.

Additionally, every timed leg is now **bracketed** by the
partition-integrity gate (entry and exit): a foreign run launching
mid-leg converts that leg into a failure instead of an undetected
contamination.

## Amendment 3 — headline rung is B37 per the divergence contingency (2026-08-19, pre-evidence for the headline rung)

The B50 divergence probe (run `20260819T083720Z-probe-3849441`, untimed,
both legs conformance-gated inside the partition) executed the
pre-registered contingency exactly: both lanes completed 50 accepted
iterations over 66 matched evaluations, but the accepted-objective
sequences **fork at iterate 38** — the matched prefix is **37** at the
1e-10 elementwise clause (cross-architecture fp64 drift accumulating
through the L-BFGS-B curvature pairs; both endpoints individually healthy,
`scipy_completed` in both lanes). Certificate cadence at 50 iterations,
recorded for the B-rationale: 2 attempts per lane (native 0.360 s, GPU
1.313 s) — the event-driven certificate cost does not scale with accepted
steps, as the Amendment-1 probe requirement was designed to verify.

Per the frozen contingency, the **headline rung is B_k\* = B37** — the
largest budget at which the probe's sequences match. No timed leg at any
headline budget has run. The anchor derives per Amendment 2a from the
B37 pairs' own matched evaluation counts × the archived sustained
per-eval mean; the B3 rung is unaffected. The B3 matrix result is also
noted here for the record: the partition sweep's best native
(omp16 SMT-assisted, 52.70 s) is **faster** than the fastest archived
unpinned-64 wall (58.702 s), so the live denominator undercuts the
archive — the anti-GPU direction the partition design promised.

## Amendment 3a — implementation record and corrections (2026-08-19, pre-verdict)

Recorded before any headline-rung timed leg completes and before any
verdict exists; nothing here changes a threshold, budget, timer, or the
pair rule.

1. **Charter-lineage validation (formalizes harness commit `23c147f32`).**
   Amendments legitimately move the charter sha mid-campaign, so each row
   binds the charter bytes **current when its run executed**, and
   `validate` accepts the append-only lineage — `92e6a657…` (freeze),
   `537d621b…` (A1), `1d82aece…` (A2), `be4b262c…` (A2a), `2dea1522…`
   (A3) — recomputing every row's contract sha against its run's own
   recorded member. The frozen sentence "charter bytes at the freeze
   commit … refuses any row whose contract sha differs from the frozen
   one" is superseded accordingly; out-of-lineage rows are still refused.
2. **Probe sequencing deviation (disclosed).** The frozen text places the
   divergence probe "immediately after Phase 1"; it actually ran after
   the B3 matrix, at the matrix-selected configuration (omp16). The probe
   remained untimed, conformance-gated, and complete before any
   headline-budget timed leg — the clause's purpose — and running it at
   the selected config makes its trajectory evidence match the config the
   pairs will use.
3. **In-child provenance, as implemented.** Captured from inside every
   lane child (sitecustomize shim, read-only, atexit): granted affinity
   mask at interpreter start and at exit, `os.cpu_count`, the threading
   and JAX env echoes (`OMP_NUM_THREADS`, `OMP_PLACES`, `OMP_PROC_BIND`,
   `OPENBLAS_NUM_THREADS`, `MKL/NUMEXPR/VECLIB`, `JAX_ENABLE_X64`,
   `JAX_PLATFORMS`), the resolved `libgomp.so.1` path from
   `/proc/self/maps`, `omp_get_max_threads()` through that handle, and
   the CPU model line. Not captured in-child, contrary to the frozen
   list, and evidenced instead at the stated level: `nvidia-smi`
   name/driver (harness partition gate + receipt-level provenance),
   `jax.__version__` (fixed by the pinned runtime env the charter names),
   and `openblas_get_num_threads()` (evidenced by the fail-closed
   `OPENBLAS_NUM_THREADS=1` echo). The receipt carries this mapping.
4. **Correction to Amendment 3's attribution.** The probe requirement and
   its certificate-cadence recording live in the frozen section "B50
   divergence probe and contingency" (shaped by the pre-freeze review),
   not in Amendment 1, which concerns only the instrument substrate.
5. **Expected timed-leg count under Amendment 2.** The frozen estimate
   (≈55–70, including disclosure legs) becomes ≈54: 15 B3-matrix +
   15 headline-matrix + 20 pair + 4 fresh-cache timed legs, still under
   the ≤96 cap; disclosure legs were retired by Amendment 2.
