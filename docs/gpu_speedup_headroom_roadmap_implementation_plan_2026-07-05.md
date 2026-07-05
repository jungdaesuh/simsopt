# GPU Speedup Headroom Roadmap: from 3.7× toward the silicon ceiling

**Status:** Draft
**Last updated:** 2026-07-05

## Purpose

Turn the 2026-07-05 headroom analysis into an executable roadmap. Current
measured state: jax-GPU beats native cpp by 3.7× (optimizer wall 695.3 s vs
2551.8 s, A100-PCIE-40GB, 255×64, mpol/ntor 10, laneC 2026-07-04) while the
raw FP64 silicon ratio between that GPU and the shared-EPYC cpp slice is
~25–50×. The gap is algorithm-shape (serial taxes), not physics. This plan
sequences the levers that close it and defines the measurements that must
precede each one.

## Goals

Frame targets as **absolute per-eval walls** (the cpp denominator is a
tunable single-node OpenMP code; ratios drift):

- Accepted eval (K1+K2) on A100: 93.8 s today (66.3+27.5, B5) → **< 40 s**
  after dense-IR, **< 15 s** after line-search batching + dispatch work.
- Trial eval on A100: 31.2 s today (B5) → **< 20 s**; line-search phase for
  a maxls=4 outer iteration ≈ **~1 eval-wall** (batched), not ~4.
- A written per-eval time attribution (≥80 % of wall accounted) BEFORE any
  tuning lever beyond dense-IR is pulled.
- Net effect: ≥8–15× vs the frozen laneC cpp baseline on the same A100;
  20–40× additionally requires H100-class FP64 + mixed precision +
  multi-GPU (kept as stretch, gated on the measurements below).

## Non-Goals

- Re-tuning the native cpp comparator (frozen as the laneC artifact).
- Changing physics tolerances or convergence semantics to buy speed.
- The dense-IR near-target solver itself — that is
  `docs/newton_near_target_dense_ir_solver_implementation_plan_2026-07-05.md`
  (Phases A/B); this roadmap consumes its outcome, does not duplicate it.
- Buying/queueing specific hardware; H100/B-class numbers are modeled until
  measured.

## Current Context (measured facts, sources)

- laneC (2026-07-04, same node): cpp 2551.8 s vs jax-GPU 695.3 s = 3.7×;
  physics parity (surface rel-diff 0.0; iota delta 4.53e-5 = documented
  native fresh-re-solve boundary).
- B5/B6 per-eval walls (A100): x0 200 s (incl. compile) + 48 s K2; trial
  31.2 s (860 pre-Newton + 50 Newton); accepted 66.3 s + 27.5 s K2;
  final-sync reuse 13.6 s.
- B6 NDJSON matvec actuals: trial Newton iters 5–23 matvecs (loose E-W);
  accepted eval `[192, 1308, 1308, 1308]` — near-target refined GMRES
  dominates accepted-eval cost. Budget formulas: single-pass 651, refined
  1302 (+1), n-independent (`optimizer.py:4396-4407`).
- System size N=1323 at mpol/ntor 10 (`optimizer.py:5133`); dense build =
  1323 HVP columns in batch-8 `lax.map` chunks (`optimizer.py:5119-5141`,
  chunk const `:3777`) ≈ one refined-GMRES budget, embarrassingly parallel.
- Cross-GPU memory (2026-06-23 code state): H100 steady ≈32 s/eval vs A100
  ≈31 s trial — near-equal walls despite 3.5× FP64 ⇒ e2e loop was
  latency/serial-bound at that algorithm shape. (Kernel-level BiotSavart is
  FP64-compute-bound per the parity-matrix-era benchmarks — both true:
  kernels are FLOP-bound, the loop is not.)
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
~35 ms/bfgs-iteration against ~0.1 ms of FP64 FLOP time at 255×64×1323
(~300× overhead, removable via fusion/command-buffers/macro-stepping, not
via more FLOPs); (3) host-driven line search — scipy evaluates trial steps
one at a time although candidate step lengths are independent (batchable).
Mixed precision and multi-GPU multiply only after those taxes are removed —
hence measurement-first ordering (design rule: never tune from intuition;
H100≈A100 equal-walls is the cautionary proof).

## Assumptions (explicit, each gets validated or falsified)

- EPYC-slice sustained FP64 for the cpp leg ~0.2–0.4 TFLOP/s (estimate,
  not measured; only used for headroom framing, no gate depends on it).
- The ~35 ms/bfgs-iteration figure attributes to in-loop kernel-launch
  chains, not host callbacks (prod runs have counters/progress off) —
  CONFIRM in M0 profiling before M3 work.
- H100≈A100 per-eval equality was measured on 2026-06-23 code; must be
  re-measured post-fidelity-fixes before any hardware conclusions.
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
   - [ ] Execute Phases A/B of the dense-IR plan (separate doc). Expected
         from this roadmap's view: accepted eval 93.8 s → ~50–60 s (Newton
         matvec chains 3×1308 → build+~3×3), churn/refinement taxes gone.
   - [ ] Record post-M1 per-eval walls (same M0 harness) — new baseline.
3. **M2 — Speculative line-search batching (~1.5–2× on optimizer wall)**
   - [ ] Scout the seam: where the decomposed lane's value/grad provider
         meets scipy L-BFGS-B's dcsrch (single_stage objective wrapper /
         `_lbfgsb_scipy.py`); design the candidate-alpha batch + cache-feed
         pattern (vmapped K1 over ≤maxls step lengths; scipy consumes from
         cache; exactness of the chosen alpha proven, not assumed).
   - [ ] Prototype behind env flag; physics gate: identical accepted step
         sequence + identical converged state vs unbatched on the iota011
         seed (byte-compare NDJSON accepted-step records).
   - [ ] Memory gate: batched K1 fits A100-40GB at chunk policy (B6 peak
         26.5 GiB single-eval → batch of 4 must stay < 40 GiB; else batch 2).
   - [ ] Measure: line-search phase wall for maxls=4 ≈ 1.0–1.3× single-eval
         wall.
4. **M3 — Dispatch-latency reduction in traced loop bodies (~up to 2× on
   bfgs-dominated phases; contingent on M0 attribution)**
   - [ ] Fusion audit of the bfgs-ondevice iteration body with
         `benchmarks/compile_breadth_probe.py` + XLA dump: kernels per
         iteration, top gap contributors.
   - [ ] Trial XLA options (command buffers / latency-hiding scheduler
         flags) on the K1 kernel; accept only flags that keep bit-identical
         results on the parity fixture.
   - [ ] If flags insufficient: macro-step batching of bfgs iterations
         (k iterations per launch) behind env knob — design-review first
         (touches the ondevice L-BFGS state machine).
   - [ ] Gate: measured ms/iteration on A100 bfgs pre-stage ↓ ≥2× with
         parity fixture bit-identical.
5. **M4 — Mixed-precision inner kernels with fp64 refinement (stretch;
   contingent on M1 landing the κ≈625-class solve paths)**
   - [ ] Scope: fp32/TF32 for HVP sweeps + GMRES/LSMR inner products ONLY
         where an fp64 IR pass certifies the final residual (m18 pattern);
         K2 dense factorization stays fp64.
   - [ ] Gate: converged states match fp64-only run within existing parity
         tolerances on the matrix; NaN-free across the production seed set;
         measured ≥1.5× on the HVP-heavy phases.
6. **M5 — Multi-GPU sharding un-pin (stretch)**
   - [ ] Produce the "multi-GPU parity/speedup proof" the BETA_QUICKSTART
         gate requires: 2-GPU hybrid-sharding run vs single-GPU — parity
         within matrix tolerances AND wall improvement > 1.3× on K1-heavy
         phases; then un-pin `jax_gpu_parity` sharding or document why not.

## Validation Plan

- [ ] Every phase: private optimizer suite green + ruff; parity fixture
      checks where the phase claims bit-identical.
- [ ] M0/M1/M2/M3 walls measured with the SAME instrument (K1 NDJSON
      per-eval events + one profiler trace), same seed (iota011), same
      salloc geometry; record in this doc's table.
- [ ] Absolute-wall targets: accepted eval < 40 s (post-M1), < 15 s
      (post-M2/M3) on A100-PCIE-40GB; trial < 20 s.
- [ ] Final: one full production-config run (maxiter 20) on A100 comparing
      total optimizer wall vs the frozen laneC artifacts; report both the
      ratio and absolute walls.

## Risks and Mitigations

- Risk: M2 batched trial evals change the accepted step (line-search
  cache-feed mismatch with dcsrch's sequential semantics).
  Mitigation: exactness gate — byte-compare accepted-step sequence;
  speculative results only ever FEED the cache, never alter the search
  logic; fall back to unbatched on cache miss.
- Risk: M2 batch × 26.5 GiB single-eval footprint OOMs the 40 GB card.
  Mitigation: memory gate before perf gate; batch size auto-derived from
  the existing byte-budget machinery; batch=2 floor.
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
- [ ] Post-M1+M2 measured: accepted < 40 s, line-search ≈ 1 eval-wall,
      A100, production seed — or a written reason the target moved.
- [ ] ≥8× vs frozen laneC cpp baseline demonstrated on one full
      production-config A100 run, with parity gates green.
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
