# GPU Speedup Headroom Roadmap: from 3.7× toward the silicon ceiling

**Status:** In progress (M1 delivered — 7.3× measured; M0/M2/M3 open;
M4 planned in its own doc; M5 stretch)
**Last updated:** 2026-07-07

## Purpose

Turn the 2026-07-05 headroom analysis into an executable roadmap. Starting
state (laneC, 2026-07-04, A100-PCIE-40GB, 255×64, mpol/ntor 10): jax-GPU
beat native cpp 4.5× on optimizer wall (529.3 s vs 2376.1 s) / 3.7× on
script totals (682.8 s vs 2547.1 s — the "3.7×" headline was the
script-total framing). Current measured state after M1 (lane B8,
2026-07-06, same config, dense-IR): **7.3× optimizer wall (324.5 s vs
2376.1 s) / 6.1× script (414.5 s vs 2547.1 s)**. The raw FP64 silicon
ratio between that GPU and the shared-EPYC cpp slice is ~25–50×; the
remaining gap is algorithm-shape (serial taxes), not physics. This plan
sequences the levers that close it and defines the measurements that must
precede each one.

## Goals

Frame targets as **absolute per-eval walls** (the cpp denominator is a
tunable single-node OpenMP code; ratios drift):

- Accepted eval (K1+K2) on A100: 93.8 s at the B5 baseline (66.3+27.5) →
  post-M1 MEASURED **67.3 s** (B7: K1 39.8 + K2 27.5; B8 maxls-4 config:
  70.8 s). The original "≤ 60 s post-M1" letter MISSED by ~12%: the
  matvec-chain collapse delivered in full (`[192, 3]`), but K1's
  bfgs-pre-stage + fixed overheads (~36 s of the 39.8 s) were never
  M1's lever — they are M3's. Post-M3 target **< 45 s** stands and is
  now sharper: it requires M3 to take K1 39.8 → ≤ 17.5 s (≥2.3× on the
  bfgs-dominated residual), on top of K2's untouched 27.5 s hard floor —
  no phase in M0–M5 touches K2 (Phase-C territory), so sub-30 s accepted
  evals are explicitly out of scope here.
- Trial eval on A100: the B5 "31.2 s" baseline is a clamped-era artifact
  (nit=50 at loose tolerances). The dense-IR-era trial does genuine
  far-from-target Newton work and MEASURES 54.9 s (B7) / 62.6 s (B8) —
  a trial-lane regression vs B5 that is the price of the fidelity fixes,
  not an M1 defect. The "< 20 s" target is SUSPENDED until M0
  attribution re-derives it; the trial lever is M2 (batching) and
  possibly M3. Line-search phase for a maxls=4 outer iteration ≈
  **~1 eval-wall** (batched, M2), not ~4. M2 moves the line-search phase
  and optimizer wall — it does not move a single accepted-eval wall.
- A written per-eval time attribution (≥80 % of wall accounted) BEFORE any
  tuning lever beyond dense-IR is pulled.
- Net effect: ≥8–15× vs the frozen laneC cpp baseline on the same A100 —
  **7.3× already measured post-M1 alone (B8, 2026-07-06)**, so the band
  floor needs only ~10% more from M2/M3; 20–40× additionally requires
  H100-class FP64 + mixed precision + multi-GPU (kept as stretch, gated
  on the measurements below).

## Non-Goals

- Re-tuning the native cpp comparator (frozen as the laneC artifact).
- Changing physics tolerances or convergence semantics to buy speed.
- The dense-IR near-target solver itself — that is
  `docs/newton_near_target_dense_ir_solver_implementation_plan_2026-07-05.md`
  (Phases A/B); this roadmap consumes its outcome, does not duplicate it.
- Buying/queueing specific hardware; H100/B-class numbers are modeled until
  measured.

## Current Context (measured facts, sources)

Primary source for the laneC/B5/B6 numbers below: the "2026-07-04
Perlmutter GPU validation campaign" close-out in
`docs/scipy_jax_decomposed_gpu_perf_gap_implementation_plan_2026-07-01.md`.

- laneC (2026-07-04, same node; artifact `crucible-gates-67bdde1a7-laneC/
  summary.json`, re-extracted 2026-07-06): cpp optimizer wall 2376.1 s /
  script total 2547.1 s vs jax-GPU 529.3 s / 682.8 s → 4.5× optimizer,
  3.7× script. (The close-out's "2551.8 vs 695.3" pairing used a
  different accounting; this doc quotes the summary.json fields.)
  Physics parity: surface rel-diff 0.0; iota delta 4.53e-5 = documented
  native fresh-re-solve boundary.
- lane B8 (2026-07-06, laneC-twin config, dense-IR on the GPU lane):
  optimizer wall 324.5 s / script 414.5 s → **7.3× / 6.1× vs the frozen
  laneC cpp artifact** (cross-run pairing, same node class; warm compile
  cache — optimizer wall insensitive). Full detail in M1 below.
- B5/B6 per-eval walls (A100): x0 200 s (incl. compile) + 48 s K2; trial
  31.2 s (860 pre-Newton + 50 Newton); accepted 66.3 s + 27.5 s K2;
  final-sync reuse 13.6 s.
- B6 NDJSON matvec actuals: trial Newton iters 5–23 matvecs (loose E-W);
  accepted eval `[192, 1308, 1308, 1308]` — near-target refined GMRES
  dominated accepted-eval cost pre-M1 (post-M1 measured `[192, 3]`, B7).
  Budget formulas: single-pass 651, refined 1302 (+1), n-independent
  (`_operator_gmres_matvec_budget`, `optimizer.py:4419`).
- K1 near-target Newton system size n=663 at mpol/ntor 10 (n≈2055 at
  mpol/ntor 18, measured 2026-07-06); κ≈625 here is κ(J), the
  well-conditioned Jacobian — the factored Newton operator is
  κ(J)²≈3.9e5, which is what M4's fp32 discussion refers to
  (`optimizer.py:5362` "measured ... at n=663, kappa=625"; the dense
  solve helpers materialize rhs-sized (`:5163`, `:5217`) and the polish
  runner passes the Newton gradient as rhs, so the build is rhs-sized;
  measured B4 dense-mode budget drop 1302→663 in the perf-gap
  close-out). Dense build = 663 HVP columns in batch-8 `lax.map` chunks
  (`_dense_square_operator_matrix`, `optimizer.py:5136`; chunk const
  `:3744`) ≈ one single-pass GMRES budget (651) ≈ HALF a refined solve
  (1302) — and embarrassingly parallel where the Krylov chain is serial.
  (The "mpol10 → N=1323" comment at `:5150` is the K2 adjoint dimension
  — the pole-1 compile-hang context — not this system. Line anchors
  re-pinned at `0a7c87040`-era HEAD; the dense-IR commit shifted them.)
- Cross-era GPU juxtaposition (two separate records, NOT one measurement):
  H100 ≈32 s/eval is the 2026-06-23 cross-GPU record (nphi127 warm steady,
  pre-fidelity-fix code; that record contains NO A100 datapoint, and it
  ALSO shows adjoint evals scaling with FP64 — A40→H100 ≈2.4×); A100
  31.2 s is the 2026-07-04 B5 trial at nphi255. Rough wall parity across a
  3.5× FP64 gap and 2× resolution is only a HINT of latency/serial
  exposure at the e2e-loop level; M0's iso-config re-measure is the
  arbiter. (Kernel-level BiotSavart is FP64-compute-bound per the
  parity-matrix-era benchmarks — a serial chain of FLOP-bound kernels can
  be latency-exposed at the loop level; the two claims are compatible but
  the loop-level one is UNPROVEN until M0.)
- fp32 campaign conclusions (2026-06-24 memories): dense-PLU on κ(H)≈κ(J)²
  ≈3.9e5 NaNs in fp32; J-based/refined systems at κ≈625 are the only true
  fp32-viable route; m18 fix = iterative refinement.
- Multi-GPU: sharding default `hybrid` (multi-device-capable) exists;
  `jax_gpu_parity` pins sharding=`none` "until a multi-GPU parity/speedup
  proof is recorded" (BETA_QUICKSTART.md device table).
- Existing probe tooling: `benchmarks/compile_breadth_probe.py`,
  `benchmarks/single_stage_outer_loop_probe.py`, K1 NDJSON per-eval events,
  matvec counters (`SIMSOPT_TRACEABLE_NEWTON_MATVEC_COUNTS`).

## Rationale

Three serial taxes hide the missing 10×: (1) Krylov chains — 651–1302
*sequential* matvecs per near-target Newton iteration (removed by dense-IR:
one batched column build + ~2 HVPs); (2) per-iteration dispatch latency —
~35 ms/bfgs-iteration against ~0.05–0.1 ms of FP64 FLOP time at
255×64×n=663 (two-plus orders of magnitude of overhead — an M0-pending
attribution, removable via fusion/command-buffers/macro-stepping, not via
more FLOPs); (3) host-driven line search — scipy evaluates trial steps
one at a time although candidate step lengths are independent (batchable).
Mixed precision and multi-GPU multiply only after those taxes are removed —
hence measurement-first ordering (design rule: never tune from intuition;
the cross-era H100/A100 wall juxtaposition is the cautionary HINT, and M0
re-measures it iso-config before it is allowed to justify anything).

## Assumptions (explicit, each gets validated or falsified)

- EPYC-slice sustained FP64 for the cpp leg ~0.2–0.4 TFLOP/s (estimate,
  not measured; only used for headroom framing, no gate depends on it).
- The ~35 ms/bfgs-iteration figure attributes to in-loop kernel-launch
  chains, not host callbacks (prod runs have counters/progress off) —
  CONFIRM in M0 profiling before M3 work.
- The H100-vs-A100 wall comparison is a cross-era, cross-config
  juxtaposition (06-23 H100 nphi127 pre-fix code vs 07-04 A100 nphi255
  post-fix) — no iso-config measurement exists; treated strictly as a
  hint until M0 measures it properly.
- Speculative line-search batching can reproduce scipy dcsrch's chosen
  step exactly (cache-feeding pattern) — needs the M2 seam scout.

## Implementation Plan

1. **M0 — Per-eval time attribution (measure first; blocks M2–M5)**
   - [ ] JAX profiler / nsys trace of ONE trial eval + ONE accepted eval on
         A100 (Perlmutter, existing checkout + `benchmark-mode`): kernel
         count, mean kernel duration, gap time, D2H sync count per eval.
   - [ ] Decompose accepted-eval 93.8 s into: pre-Newton bfgs, Newton
         (matvec chains), K2 adjoint, host/sync/progress overhead. Written
         table, ≥80 % attributed.
   - [ ] Re-measure H100-vs-A100 per-eval walls at current HEAD (one
         RunPod/H100 B-lane rerun) to refresh the latency-bound evidence.
   - [ ] Gate: attribution doc appended to this plan; M2/M3 priorities
         re-ranked from data (their estimates below are provisional).
2. **M1 — dense-IR near-target solver (delegated)**
   - [~] Execute Phases A/B of the dense-IR plan (separate doc). Phase A
         v1 SHIPPED `ad3cc28b7` (2026-07-05) and validated on local CPU:
         near-target iteration pattern measured `[189, 3]` — one loose
         E-W operator iteration (unchanged by design) + one 3-matvec IR
         iteration with the 663-column build uncounted; eval-1 K1
         799.3 s success vs hybrid 1552.2 s vs operator REJECTED.
         This roadmap's pre-measurement estimate was accepted eval
         93.8 s → ~50–60 s (Newton matvec chains 3×1308 → one
         663-column build + ~3×3; K2's 27.5 s untouched) — SUPERSEDED
         by the measured 67.3 s in the next item (the estimate
         under-counted K1's bfgs/fixed residual, which is M3 scope).
         Phase B (self-deciding default) still open.
   - [x] Record post-M1 per-eval walls — MEASURED (lane B7, job
         55547957, 2026-07-05, A100, B5-twin config): accepted eval
         **93.8 → 67.3 s** (K1 66.3 → 39.8 s with actuals
         `[192, 3]`; K2 27.5 s unchanged = 41% of the accepted eval,
         the M3 target); trial 54.9 s (honest reject); x0 242 s incl.
         dense-IR fresh compile; total 613.0 vs B5 644.7 s; ‖grad‖
         1.81e-15. M1 mechanism fully delivered; remaining accepted-
         eval headroom belongs to M2 (line-search batch) and M3 (K2).
   - [x] **Post-M1 cpp headline — MEASURED (lane B8, job 55559869,
         2026-07-06, laneC/A5b config maxiter 20 / maxls 4 /
         255×64): GPU-vs-native-cpp optimizer wall 2376.1 → 324.5 s
         = 7.3× (was 4.5× with operator GMRES, A5b 528.7 s);
         script totals 2547.1 → 414.5 s = 6.1× (was 3.7×).**
         Same bfgs path as A5b legs (pre=701, shared seed spec);
         accepted eval 70.8 s (K1 43.4 `[180, 3]` grad 1.51e-14 +
         K2 27.4); converged same basin (Vol 0.049164,
         Iota 0.110198). Caveats: cpp pairing is cross-run
         (laneC, same node class); warm compile cache (optimizer
         wall insensitive; cold adds ~1-2 min to script total).
3. **M2 — Speculative line-search batching (~1.5–2× on optimizer wall)**
   - [ ] Scout the seam: where the decomposed lane's value/grad provider
         meets scipy L-BFGS-B's dcsrch (single_stage objective wrapper /
         `_lbfgsb_scipy.py`); design the candidate-alpha batch + cache-feed
         pattern (vmapped K1 over ≤maxls step lengths; scipy consumes from
         cache; exactness of the chosen alpha proven, not assumed).
   - [ ] Prototype behind env flag; gates run under the deterministic
         `*_parity` backend mode — production lanes make no byte-identity
         claim (jax_parity_status.md), and vmapped batching changes
         reduction structure regardless. Gate = identical accepted-ALPHA
         sequence + tolerance-based converged-state equality vs unbatched
         on the iota011 seed; byte-compare is a parity-mode diagnostic
         only, never the gate.
   - [ ] Memory gate, model-first: the B6 whole-run peak (26,469 MiB ≈
         25.8 GiB) is dominated by the ACCEPTED-eval near-target dense
         build, which trial evals do not run — the shared-vs-per-element
         split of a batched-trial footprint is unknown. Measure a batch-2
         trial K1 first, derive the per-element delta, and set the batch
         cap from that measurement. No batch size is promised from
         arithmetic on the 25.8 GiB figure.
   - [ ] Measure: line-search phase wall for maxls=4 ≈ 1.0–1.3× single-eval
         wall.
4. **M3 — Dispatch-latency reduction in traced loop bodies (~up to 2× on
   bfgs-dominated phases; contingent on M0 attribution)**
   - [ ] Fusion audit of the bfgs-ondevice iteration body with
         `benchmarks/compile_breadth_probe.py` + XLA dump: kernels per
         iteration, top gap contributors.
   - [ ] Trial XLA options (command buffers / latency-hiding scheduler
         flags) on the K1 kernel — this EXECUTES the command-buffer /
         CUDA-graph A/B already scoped (with a null-result exit) in the
         perf-gap plan; not new scope. Flags that keep the parity fixture
         bit-identical are accepted directly; flags that reorder
         reductions get the predecessor's parity-tolerance A/B treatment
         and a null-result exit.
   - [ ] If flags insufficient: macro-step batching of bfgs iterations
         (k iterations per launch) behind env knob — design-review first.
         NAMED PRIOR: the 422 GiB ondevice compile blowup was graph
         BREADTH, and the torax-style host-controlled kernelization plan
         warns a stepwise driver "can still be an expensive macro-kernel
         when each per-step kernel encloses the full pipeline"; its
         companion pole2 compile-breadth plan carries the recorded
         macro-step breadth-exclusion gate. Any macro-step design must
         pass a compile-breadth probe before it touches the ondevice
         L-BFGS state machine.
   - [ ] Gate: measured ms/iteration on A100 bfgs pre-stage ↓ ≥2× with
         parity fixture bit-identical.
5. **M4 — Mixed-precision inner kernels with fp64 refinement (stretch;
   requires NEW scope that no earlier phase delivers)**
   → SSOT for M4 scope is now
   `docs/mixed_precision_upgrade_implementation_plan_2026-07-07.md`
   (P0 dtype scout → P1 fp32 factors in dense-IR → P2 fp32 loose phase
   w/ E-W handoff → P3 LSMR-IR adjoint → P4 fp32 kernels → P5 RTX-5090
   validation), driven by the RTX 5090 32 GB / FP64-1:64 user
   requirement. The two scoping bullets this section previously carried
   are superseded there: the "J-based-solve-first or does not begin"
   precondition is retired by the IR theory check (κ(H)·2⁻²⁴ ≈ 0.023 < 1
   ⇒ the SHIPPED dense-IR backward-error gate certifies fp32 FACTORS on
   the κ² system directly = P1; the recorded 2026-06-24 fp32 NaNs were
   whole-solve fp32, a different regime), and the J-based κ≈625 LSMR
   route remains the true-fp32 endgame as P3 (lineax LSMR already wired,
   `solve/dispatch.py:517`).
   - [ ] Execute the mixed-precision plan P0–P4; report per-phase gates
         there. Gate mirrored here: converged states match the fp64-only
         run within existing parity tolerances on the matrix; NaN-free
         across the production seed set; measured ≥1.5× on the HVP-heavy
         phases; fp64 mode byte-identical (P0 hash gate).
6. **M5 — Multi-GPU sharding un-pin (stretch)**
   - [ ] Produce the "multi-GPU parity/speedup proof" the BETA_QUICKSTART
         gate requires: 2-GPU hybrid-sharding run vs single-GPU — parity
         within matrix tolerances AND wall improvement > 1.3× on K1-heavy
         phases; then un-pin `jax_gpu_parity` sharding or document why not.
         RECORDED PRIOR (2026-06-23 cross-GPU campaign): multi-GPU bought
         THROUGHPUT (3.79× on 4×H100 across independent configs), not
         single-solve latency, with no memory pressure forcing a shard —
         so this proof attempt tests single-solve sharding against a
         recorded null expectation, and throughput-parallel independent
         configs are the already-proven fallback win.

## Validation Plan

- [ ] Every phase: private optimizer suite green + ruff; parity fixture
      checks where the phase claims bit-identical.
- [ ] M0/M1/M2/M3 walls measured with the SAME instrument (K1 NDJSON
      per-eval events + one profiler trace), same seed (iota011), same
      salloc geometry; record in this doc's table.
- [x] Post-M1 accepted-eval wall MEASURED: 67.3 s (B7; the "≤ 60 s"
      letter missed ~12% — adjudicated in Goals: M1's lever delivered
      fully, the residual is M3 scope).
- [ ] Absolute-wall targets on A100-PCIE-40GB still open: accepted eval
      < 45 s (post-M3 ⇒ K1 39.8 → ≤ 17.5 s); trial target SUSPENDED
      pending M0 re-derivation (measured 54.9–62.6 s dense-IR-era; the
      B5 31.2 s baseline was a clamped-era artifact); line-search phase
      ≈ one eval-wall (post-M2). K2's 27.5 s floor stands until Phase-C
      scope.
- [ ] Final: one full production-config run (maxiter 20) on A100 comparing
      total optimizer wall vs the frozen laneC artifacts; report both the
      ratio and absolute walls.

## Risks and Mitigations

- Risk: M2 batched trial evals change the accepted step (line-search
  cache-feed mismatch with dcsrch's sequential semantics).
  Mitigation: exactness gate under the deterministic `*_parity` mode —
  identical accepted-alpha sequence + tolerance-based state equality;
  speculative results only ever FEED the cache, never alter the search
  logic; fall back to unbatched on cache miss.
- Risk: batched trial K1s OOM the 40 GB card (per-element footprint
  unknown until measured).
  Mitigation: model-first memory gate — measure batch-2, derive the cap;
  no batch size promised before that measurement.
- Risk: M3 XLA flags perturb reduction order → parity drift.
  Mitigation: accept only bit-identical-on-fixture flags; otherwise treat
  as M4-class (tolerance-gated) and defer.
- Risk: Amdahl — post-M1 the fixed per-eval costs (K2 small dense solves,
  host syncs, progress I/O) dominate and M2–M4 underdeliver e2e even if
  their phase-local gates pass.
  Mitigation: M0 attribution re-run after each phase; stop pulling levers
  when the residual is host-fixed cost; that residual defines the honest
  ceiling and gets recorded, not fought.
- Risk: the H100≈A100 equality was an artifact of the 06-23 code state and
  misleads prioritization.
  Mitigation: M0 re-measures it at HEAD before any hardware framing.

## Completion Criteria

- [ ] M0 attribution table in this doc; priorities re-ranked from it.
- [ ] Post-M1..M3 measured: accepted < 45 s, line-search ≈ 1 eval-wall,
      A100, production seed — or a written reason the target moved.
- [ ] ≥8× vs frozen laneC cpp baseline demonstrated on one full
      production-config A100 run, with parity gates green. CURRENT
      POSITION: 7.3× measured on exactly such a run (B8, maxiter 20,
      2026-07-06) — M1 alone; the remaining ~10% belongs to M2/M3.
- [ ] Stretch phases (M4/M5) either landed with their gates or explicitly
      parked with measured justification.

## Open Questions

- M2 seam: does the decomposed provider see raw alphas from dcsrch early
  enough to speculate, or only one candidate at a time? (Owner: M2 scout;
  determines batching pattern.)
- Does `--target-lane-accepted-step-sync per-accept` interact with batched
  trials (accepted-step replay uses trial-cache state)? (Owner: M2 scout.)
- Which XLA version knobs exist in jax/jaxlib 0.10.0 for command buffers
  on CUDA (vs needing an upgrade)? (Owner: M3; check before promising.)
- H100 access path for M0's re-measure: RunPod (egress/driver traps per
  runbook) vs wait for Perlmutter A100-only. (Owner: user preference.)
