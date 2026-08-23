# JAX GPU examples backlog — native-speed implementation plan

**Status:** Done (probe phase; certified campaigns are chartered follow-ups)
**Last updated:** 2026-08-23

## Purpose

Convert the 2026-08-23 theoretical-ceiling classification of `examples/jax`
mirrors into executable work: engineering levers, diagnostic A/B probes, and
charter drafts for the mirrors that can plausibly beat native C++/simsoptpp on
GPU but have no receipt today. The classification found, over the 27 native
mirrors: 2 certified GPU wins, ~8 plausible, 4 marginal, 13 structural-no.
This plan owns the plausible and marginal sets — with one carve-out: the
eighth plausible mirror (the single-stage nested track) is executed elsewhere
under its own live charter and is only tracked here (§Phase 5). The authority
for "where to launch an example today" remains `docs/jax_example_device_assignment.md`
(4 gpu / 25 cpu / 11 unmeasured rows); this plan changes that record only
through its amendment procedure.

**Claim discipline (binding for every phase).** This plan mints no speed
claim. Probes are diagnostic-not-certifying by construction. A certified
native-vs-GPU claim requires its own preregistered charter frozen before
evidence (five-pair interleaved rule, swept-OMP fair-native denominator,
warm/cold scoping, oracle-verified endpoints) per §Campaign protocol. Charter
drafts produced here are deliverables; running them to a terminal receipt is
gated follow-up work, not a task in this plan.

## Goals

- Every plausible/marginal mirror in scope ends with exactly one of: a
  diagnostic probe number at the decisive scale, or a named blocker recorded
  in this document.
- Engineering levers landed with tests: the `permanent_magnet_simple` mirror
  `native_default` scale fix, the stochastic sample-axis batching lever, and
  probe harnesses under `benchmarks/`.
- A charter draft exists for each family whose probe clears ≥1.10× warm
  against swept-OMP native.
- `docs/jax_example_device_assignment.md` rows amended only per its procedure
  (tracked receipt required for any `gpu` move; append-only log).

## Non-Goals

- **No edits to native examples' shipped defaults.** The native example is
  the oracle and the denominator; off-default-scale comparisons run as
  `benchmarks/`-path harnesses that build the same configuration through
  public APIs in both lanes (precedent: `benchmarks/stage_two_finitebuild_native_gpu.py`).
- **No cold-start claims, no bounded-scale timing evidence, no claims minted
  here.** Bounded runs are launch-bound artifacts (`docs/jax_example_device_assignment.md`,
  "Bounded-scale timings are excluded by construction").
- **The nested-LS outer track is fenced.** `single_stage_boozer_vacuum_optimization`
  is owned by the live eight-term outer charter (`docs/jax_nested_ls_outer_charter.md`,
  amendment lineage through Amendment 3, commit `4f0b8bdbd`) executed by a
  concurrent session. This plan tracks it (§Phase 5) and touches none of its
  files: `benchmarks/nested_ls_outer_*.py`, `docs/jax_nested_ls_outer_charter.md`,
  `src/simsopt_jax_adapters/geo/nested_ls_contract.py`,
  `src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py`,
  `tests/geo/test_nested_ls_reduced_scale.py`. Cloning the *pattern* of the
  thin probe scripts `benchmarks/nested_ls_a100_banana_omp.py` /
  `benchmarks/nested_ls_f3_b37_gpu_canaries.py` is allowed; editing them is not.
- Never-winnable rows (3 tracing, `boozer`, `boozerQA`, `qfm`,
  `stage_two_optimization_minimal`, `wireframe_rcls_basic`, 4 tiny mirrors,
  VMEC hybrid, 10 toy tutorials) are out of scope; their mechanism rows stand.

## Current Context

Facts verified 2026-08-23 in this worktree (branch `pr/jax-port-squashed`;
the concurrent nested-LS session moves this repo daily — re-verify
file:line anchors before acting on them):

- **Certified wins to imitate:** `wireframe_gsco_multistep` 3.5× warm device
  solve (`docs/receipts/wireframe_gsco_multistep_native_default_receipt.md`);
  `stage_two_optimization_finitebuild` 13.58× warm solve / 3.11× warm
  persistent-cache wall, cold 0.88× bounded negative
  (`docs/receipts/stage_two_finitebuild_native_gpu_successor.md`); flat675
  fused 7.70×@B37 (`docs/receipts/flat675_fused_campaign.md`).
- **Mechanism law** (`docs/jax_example_device_assignment.md`): wide batched
  work with large reductions → GPU; narrow sequential chains and tiny
  problems → CPU; reduction dimension × per-step work volume decides.
  Empirical threshold from the measured points: ≳10⁷ per-step reduction
  elements or ~1 GFLOP/eval.
- **The fused on-device L-BFGS driver is already the default fast-lane
  path** for the stochastic and stage-two-family mirrors — not an untapped
  lever. `Driver.SIMSOPT_LBFGSB` → `dispatch.minimize` →
  `_legacy_lbfgsb_options` (`src/simsopt_jax/solve/dispatch.py:161-176`)
  auto-selects `lbfgs_run_mode="fused_stepwise"` whenever no step observer is
  attached, landing in the same on-device `lax.while_loop` kernel
  (`src/simsopt_jax/geo/optimizers/private/_lbfgs.py:442-527`) that
  finite-build/flat675 time through `fused_lane.solve_fused_lane`. It is
  objective-generic (`value_and_grad_fn`, `x0`, `SimsoptLBFGSBOptions`;
  `src/simsopt_jax/solve/simsopt/contracts.py:11-17`). One disclosed
  overhead: `serial_solve_jax` writes a bounded-objective log per solve
  (2 extra host materializations per solve, not per iteration;
  `src/simsopt_jax/solve/serial.py:629-637`).
- **GSCO siblings (U1):** shipped 48×50/2,000-it lose 0.79–0.89× vs best
  native. The native sources name reference values in dead comments
  (`examples/2_Intermediate/wireframe_gsco_modular.py:27,30,33` —
  96/100/20,000; sector-saddle additionally `break_width` 2→4,
  `gsco_cur_frac` 0.05→0.03, `lambda_S` 10⁻⁶·⁵→10⁻⁷·⁵ at `:58,:61,:69`).
  No CLI/env selector exists in either script. At reference scale the
  reduction shape is 1,024×19,200 — identical to the certified multistep win
  (segments ≈ 2·nφ·nθ, `src/simsopt/geo/wireframe_toroidal.py:123`).
  Caveat: reference siblings run 20,000 iterations in one flat stage vs
  multistep's 7×2,500 over shrinking masked subsets — same per-step shape,
  different aggregate structure; the probe decides. Both lanes share the
  per-candidate GSCO math (`src/simsopt_jax/core/wireframe_workflow.py:650-663`)
  but different outer drivers (siblings:
  `_greedy_stellarator_coil_optimization_sampled_jax` `:815`; multistep:
  `_gsco_live_loop_unchecked` `:772`). The sibling campaign harness is
  host-local only; in-tree there are just the three multistep capture scripts
  (`docs/receipts/wireframe_gsco_multistep/`), hardcoded to multistep.
- **Stochastic stage-two (U3):** the only family whose *shipped* scale sits
  above the mechanism threshold: 1,024 surface pts × 16 coils × 360 quad
  × 16 samples = **94.4M pair-evals per objective eval**
  (`src/simsopt_jax/examples/stochastic_stage_two.py:46-69`). The uncaptured
  lever is the sequential sample loop:
  `jax.lax.scan(jax.checkpoint(...))` at
  `src/simsopt_jax/objectives/stochastic_stage_two.py:64-68` — no sample-axis
  parallelism today. Memory (fp64, per objective eval): scan ≈ 0.43 GB
  resident (one sample), full vmap over 16 ≈ 6.8 GB, tile=8 ≈ 3.4 GB,
  tile=4 ≈ 1.7 GB; the 256-sample out-of-sample lane must stay scanned
  (full vmap there ≈ 109 GB). Bitwise shared-sample plumbing already exists:
  the parity case materializes perturbations once and injects the same fp64
  bytes into both lanes with no MPI
  (`examples/jax/parity/cases/native_stage_two_optimization_stochastic.py:124-172,283-319`).
  Matched-work traps: native uses `MPIObjective(..., needs_splitting=True)`
  (`examples/2_Intermediate/stage_two_optimization_stochastic.py:142`) so the
  denominator is two-dimensional (MPI ranks × OMP); native `maxcor=400`
  (`:198`) vs mirror `maxcor=10`; native runs a 5-ε Taylor test (~10 evals,
  `:188-191`) and a 256-sample OOS loop (`:219-229`) outside the comparable
  solve.
- **PM family (U2):** every GPMO variant the mirrors use runs on-device via
  `lax.scan` with a 1:1 native binding (`src/simsopt_jax/core/pm_optimization.py:886,2061`;
  `src/simsoptpp/python.cpp:74-78`). The done-skip (`lax.cond` frozen step)
  exists **only** in ArbVec-backtracking (`pm_optimization.py:1971,2042-2052`)
  — so the measured MUSE 4.05× loss (device-assignment row, "dated
  pre-cond-skip") is stale evidence for MUSE/PM4Stell and was never
  applicable to the baseline kernel. Mirror scales match native shipped
  scale for MUSE/PM4Stell/QA (nφ=16, downsample=10, matched iteration
  budgets), and all three native sources name "≥ 64 for real runs".
  **`permanent_magnet_simple` mirror is a real defect:** it hardcodes
  nφ=nθ=2 / downsample=100 and its `solve()` ignores `_scale`
  (`examples/jax/1_Simple/permanent_magnet_simple.py:49-50,74,80`), i.e. its
  `native_default` equals the native **CI** branch (4 rows / 574 dipoles vs
  native non-CI 256 rows / 14,336 dipoles) — while its own parity case
  already carries the correct values (nφ=16, downsample=4, K=500;
  `examples/jax/parity/cases/native_permanent_magnet_simple.py:35-38`).
  Memory at nφ=64 fits the 5090 comfortably: MUSE peak ≈ 3.3 GB (A_obj
  740 MB + contributions 740 MB + connectivity transient 1.36 GB, ndipoles
  7,530), PM4Stell ≈ 6.8 GB (27 polarizations, ndipoles 5,826); the
  `(N,N)` connectivity tensors are the binding constraint only if
  `downsample` drops below 10. Bookkeeping mismatch to equalize in any
  matched-work probe: mirrors record 1 history snapshot
  (`record_every=max_steps`) vs native nhistory 10–500. QA's mirror uses
  relax-and-split (`relax_and_split_jax`,
  `src/simsopt_jax/solve/permanent_magnet.py:919` → MwPGP inner solve) — a
  kernel exists; its ndipoles at nφ=64 is unmeasured (grid built per-φ-plane).
  No PM benchmark harness or receipt exists in-tree; the parity cases'
  `_scale_configuration()` is the natural place for an nφ=64 rung.
- **Marginal quartet (U5):** per-eval Biot-Savart pairs: `stage_two_optimization`
  and `_planar_coils` 16 coils × 100 quad × 1,024 pts = **1.64M**;
  `coil_forces` 12 × 75 × 1,024 = **0.92M** plus a force term of only
  ≈ 202,500 pair terms (force evaluated for the 3 base coils, not all 12;
  `src/simsopt_jax_adapters/field/force.py:2006-2093`,
  `examples/jax/3_Advanced/coil_forces.py:75`). All three already run the
  fused on-device kernel (see above) — the probe times what exists.
  `wireframe_rcls_with_ports` device solve is a null-space
  equality-constrained least squares, not a dense normal-equations solve:
  complete QR of `Cᵀ` (497×254) then `jnp.linalg.lstsq` on a 1,521×243
  reduced system (`src/simsopt_jax_adapters/solve/wireframe.py:150,172-176,296-352`;
  n_segments 528 → 497 free after port removal, 254 constraints, 1,024
  plasma rows). The n≈169/716 crossover numbers on its device row are
  host-local provenance; this specific solve shape has never been timed.
- **Nested-LS outer status (U4):** charter at
  `docs/jax_nested_ls_outer_charter.md`; FD-0 ran and **failed closed**
  (`fail_closed_reason="fd_rel_error_unmet"`, 3/11 directions,
  `docs/receipts/evidence/nested_ls_outer_fd0_20260823.amendment2-red.*`);
  Amendment 3 (committed `4f0b8bdbd`) replaces the FD ladder; next steps are
  the canonical FD-0 rerun, then B3 (swept-OMP artifact-bound), then B37. **Resolved later on 2026-08-23: the rerun landed GREEN 11/11 (`484b3fc26`); B3/B37 remain.**
  The charter mints a **benchmarks-path claim only** — it never names the
  example; example-level promotion would be a separate F4-style step
  (`docs/jax_flat675_promotion_plan.md` precedent), and the outer
  instrument's eight-term J is not the shipped example's configuration
  (`examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py`
  uses `surface_vessel_weight: 0.0` and `minimize_lbfgs_host_core`).

## Campaign protocol (extracted from the governing docs — cite, don't restate)

Any charter draft produced by this plan must carry, with these citations:

1. **Preregistration**: new campaign = new preregistration with its own gate
   derivation; amendments dated, pre-evidence only, append-only
   (`docs/jax_gpu_finitebuild_native_speed_successor_plan.md:7-8,209,247-251`;
   `docs/jax_gpu_flat675_fused_campaign_plan.md:267-271,293-294`).
2. **Win rule**: five interleaved pairs, alternating order; median paired
   `native/gpu ≥ 1.10` with every pair `> 1.00`
   (`…successor_plan.md:198-206`). F3 additionally binds a dual anchor rule
   (`…flat675_fused_campaign_plan.md:226-245`) — adopt when an archived
   anchor number exists.
3. **Fair-native denominator**: sweep `OMP_NUM_THREADS` over
   `{2,4,8,16,32,48}` (and optimizer history where applicable); the
   denominator is the swept optimum, the shipped default disclosed
   separately; callback cost charged to native
   (`…successor_plan.md:186-189`;
   `docs/receipts/stage_two_finitebuild_native_gpu_successor.md:40-42`;
   sweep-down correction: `docs/receipts/stage_two_minimal_coupled_route.md:574-583`).
   For the stochastic family the denominator is two-dimensional: MPI ranks ×
   OMP, both pinned and disclosed.
4. **Warm/cold scoping**: warm same-process and warm persistent-cache claims
   only; cold disclosed under the same rule, never folded in; lever =
   `JAX_COMPILATION_CACHE_DIR`
   (`benchmarks/stage_two_finitebuild_native_gpu.py:1189,1286`).
5. **Physics gate**: endpoints oracle-verified through the native evaluator;
   bitwise where the algorithm is exact greedy arithmetic (GSCO/GPMO);
   tolerance buckets otherwise (`src/simsopt_jax/parity_tolerances.py`).
6. **Provenance**: per-leg identity JSON (commit + dirty sha, observed OMP,
   jax platform/x64/devices, nvidia-smi processes), cross-row conformance
   fail-closed (`benchmarks/stage_two_finitebuild_native_gpu.py:474-517,3568+`).
7. **Receipt + scoreboard**: terminal receipt tracked under `docs/receipts/`,
   evidence bundle under `docs/receipts/evidence/<campaign>/`, device row
   moved in the same commit as an append-only log entry; enforced by
   `tests/test_jax_example_device_assignment.py:248-331` (tracked-receipt,
   evidence-class, and mechanism-family gates).

## Rationale

Priority order is expected-value order. (P1) stochastic stage-two is the only
family whose *shipped* scale already sits above the mechanism threshold, with
the certified finite-build campaign as the harness template and the
shared-sample parity plumbing already built — and its one uncaptured lever
(the sample-axis scan) is a scoped, testable change. (P2) GSCO siblings need
no kernel work at all — only a benchmarks-path harness at the reference
configuration their own sources name; cheapest probe per unit of expected
win. (P3) the PM family combines a genuine mirror defect (pm_simple scale), a
stale negative worth retiming (MUSE post-cond-skip), and a documented
"real-run" resolution (nφ=64) that clears memory checks. (P4) marginal
quartet gets kill-fast probes only — their fused drivers are already live, so
a probe is pure measurement. (P5) is tracking-only. Alternatives considered:
editing shipped native example defaults to reference scale (rejected —
changes the claim target and the one-to-one mirror contract); building one
generic campaign harness for all families (rejected — the certified pattern
is thin per-family probe scripts delegating to library functions, e.g.
`benchmarks/nested_ls_a100_banana_omp.py`, 86 lines; a premature generic
harness would be a shallow module).

## Assumptions

- The RTX 5090 box (this machine) is available for probes; A100 (landau) is
  optional cross-check, not required by any task here.
- File:line anchors above were verified 2026-08-23 but the concurrent session
  commits daily; each phase re-verifies its anchors before editing.
- Probe budgets may be reduced from native-default iteration counts only for
  per-iteration-rate measurements; any warm-solve ratio quoted in a probe
  outcome uses matched work on both lanes.

## Implementation Plan

Task tags: `[eng]` code/tests, `[probe]` diagnostic measurement, `[charter]`
charter-draft document, `[doc]` bookkeeping. Every `[probe]` script
self-labels `diagnostic-not-certifying`, interleaves lanes, records per-leg
identity JSON, and writes its artifact under
`docs/receipts/evidence/` (thin-script precedent:
`benchmarks/nested_ls_a100_banana_omp.py`).

### Phase 0 — Shared probe conventions

- [x] `[eng]` P0.1 (2026-08-23, commit 76f1b5f37; scope grew during review:
      scrub-then-pin environment builder, leg ledger, and public digest/ULP
      helpers joined the module) `benchmarks/probe_conventions.py`: the shared helpers the
      family probes need — identity-JSON writer (fields cloned from
      `_runtime_identity`, `benchmarks/stage_two_finitebuild_native_gpu.py:474-517`),
      interleave scheduler, OMP-sweep runner over `{2,4,8,16,32,48}`,
      warm/cold leg wrapper honoring `JAX_COMPILATION_CACHE_DIR`, and the
      `diagnostic-not-certifying` stamp. No campaign logic, no gates.
      Unit test under `tests/benchmarks/`.

### Phase 1 — `stage_two_optimization_stochastic` (likeliest shipped-scale win)

- [x] `[probe]` P1.1 (2026-08-23 — warm GPU 1.35×/1.24× at maxcor 10/400, matched nit=400, cold 157–214 s; see §Probe outcomes) Baseline probe, shipped scale (16 samples, matched
      budget, **matched policy**): both lanes at the same `maxcor` — one
      matched pair set at 10 (mirror default,
      `examples/jax/2_Intermediate/stage_two_optimization_stochastic.py:263`)
      and one at 400 (native default,
      `examples/2_Intermediate/stage_two_optimization_stochastic.py:198`) — so
      policy and hardware are never mixed in one ratio. Native additionally
      swept over OMP `{2,4,8,16,32,48}` at 1 MPI rank (rank count pinned and
      recorded), interleaved, N=3 pairs, warm+cold GPU legs.
      Shared-sample injection via the parity-case plumbing so both lanes
      consume identical perturbation bytes; Taylor-test and OOS stages
      excluded from the timed window on both lanes symmetrically.
- [x] `[eng]` P1.2 (2026-08-23, commit c3a827510) Sample-axis lever: tiled scan-of-vmap over the sample axis
      in `stochastic_flux_mean_from_geometry`
      (`src/simsopt_jax/objectives/stochastic_stage_two.py:64-68`), tile size
      swept `{4,8,16}`, default from measurement; the 256-sample
      out-of-sample lane stays scanned (109 GB full-vmap is out of reach).
      Parity test vs the scan path at the `native_workflow` tolerance bucket
      — bit-identity is NOT assumed because the sample-sum reduction order
      changes; record the max diff. Keep the scan path as the oracle.
- [x] `[probe]` P1.3 (2026-08-23 — tiles measured NEUTRAL-to-negative: 24.94–25.21 s vs 24.62 s untiled at budget 400; no winning tile exists, the fused scan lane is already the production configuration) Re-probe P1.1 with the winning tile; record the ratio
      table (warm solve, process wall, per-eval marginal) in §Probe outcomes.
- [x] `[charter]` P1.4 (2026-08-23 — bar cleared at 1.24–1.35×; draft written, incl. a coil-forces rung) charter draft
      `docs/jax_gpu_stochastic_stage_two_campaign_plan.md` per §Campaign
      protocol — includes the matched-sample identity gate
      (fingerprints from `src/simsopt_jax/examples/stochastic_samples.py:98-130`),
      the two-dimensional ranks×OMP denominator law, and the harness-clone
      gaps U3 named (retarget `_parity_case`, add `build_native_evaluator`
      to the stochastic parity case, add `objective_scale`).

### Phase 2 — GSCO siblings at reference scale

- [x] `[eng]` P2.1 (2026-08-23, commit 76f1b5f37) `benchmarks/wireframe_gsco_siblings_reference_scale.py`:
      builds the reference configuration (96×100, 20,000 it; sector-saddle
      `break_width=4`, `gsco_cur_frac=0.03`, `lambda_S=10**-7.5`; plasma 32²
      unchanged) through the same public APIs in both lanes — native
      `optimize_wireframe(..., 'gsco', ...)` vs JAX `gsco_wireframe_jax`
      (`src/simsopt_jax_adapters/solve/wireframe.py:241`). Reuse the capture
      pattern of `docs/receipts/wireframe_gsco_multistep/`; do not edit the
      example scripts.
- [x] `[probe]` P2.2 (2026-08-23 — see §Probe outcomes: 5.2×/4.4× WINs, bitwise) Interleaved A/B at reference scale per sibling: warm
      device solve vs swept-OMP native; full-precision final segment-currents
      comparison (bitwise expected — GSCO is exact greedy arithmetic; the
      siblings receipt proved 0 ULP at shipped scale).
- [x] `[probe]` P2.3 (2026-08-23 — GPU per-iteration near-flat 0.20→0.26 ms/it; caveat dissolved) Quantify the aggregate-structure caveat: per-iteration
      device time and native per-iteration time at both 48×50 and 96×100, so
      the flat-20,000-iteration vs staged-17,500 difference is measured, not
      argued.
- [x] `[charter]` P2.4 (2026-08-23 — both siblings cleared at 5.2×/4.4×; draft written incl. the shipped-scale conflict re-adjudication rung) charter draft
      `docs/jax_gpu_gsco_siblings_reference_scale_campaign_plan.md`, claim
      scoped to the reference configuration by name (the shipped-scale
      `cpu` rows stand unless separately overturned).

### Phase 3 — Permanent-magnet family (GPMO / relax-and-split)

- [x] `[eng]` P3.1 (2026-08-23, commit f37c86350; the row moved to
      `unmeasured` rather than staying `cpu` — review found the `cpu`
      placement contradicted the record's own coherence rule) Fix the
      `permanent_magnet_simple` mirror `native_default`
      branch: align to its parity-case SSOT (nφ=nθ=16, downsample=4, K=500;
      `examples/jax/parity/cases/native_permanent_magnet_simple.py:35-38`),
      branching on the scale argument its `solve()` currently ignores
      (`examples/jax/1_Simple/permanent_magnet_simple.py:49-50,74,80`);
      bounded scale keeps the current 2×2/100 values. Parity + example tests
      at both scales; manifest smoke args unchanged. Same commit: amend the
      `native-permanent-magnet-simple` row's mechanism text in
      `docs/jax_example_device_assignment.md:183` per the amendment
      procedure — "fixed 2x2 quadrature grid … no native_default branch"
      becomes stale; the row stays `cpu` and re-opens with the `narrow
      matrix` family (256-row reduction) until a probe says otherwise.
- [x] `[probe]` P3.2 (2026-08-23 — 5.2× WIN, bitwise; supersedes the 2026-07-26 ~2× diagnostic) `permanent_magnet_simple` matched-work A/B at native
      non-CI scale (16×16, downsample=4, K=500, GPMO baseline): swept-OMP
      native vs warm+cold GPU, interleaved, bitwise moments check —
      supersedes the 2026-07-26 N=3 diagnostic (~2.0×) under probe
      conventions. History bookkeeping equalized (native `nhistory` vs
      mirror `record_every`) before timing.
- [x] `[probe]` P3.3 (2026-08-23 — 0.64×: the 4.05× was stale; still CPU-favored at nφ=16) MUSE shipped-scale retime (nφ=16, matched work,
      ArbVec-backtracking with the now-landed frozen-step skip,
      `src/simsopt_jax/core/pm_optimization.py:1971,2042-2052`): directly
      tests whether the 4.05× loss on the device row is stale even at
      shipped scale.
- [x] `[probe]` P3.4 (2026-08-23 — MUSE 2.9× WIN bitwise; PM4Stell 3.0× speed but BLOCKED on the k=201 greedy fork, dumps archived) MUSE + PM4Stell at nφ=nθ=64 ("real run" resolution named
      in the native sources), matched iterations, matched algorithm variant,
      matched history bookkeeping. Memory pre-cleared by U2 estimates
      (≈3.3 GB / ≈6.8 GB fp64 on a 32 GB 5090); keep `downsample=10` — the
      `(N,N)` connectivity tensors bind if it drops.
- [x] `[probe]` P3.5 (2026-08-23 — grid 29,286 dipoles / 2.88 GB fits; native RS ~32.4 s OMP-insensitive; GPU lane BLOCKED fail-closed: the native alpha formula exceeds the JAX MwPGP 2/λ_max bound by ~1e-4 — named charter adjudication item) `permanent_magnet_QA` (relax-and-split → MwPGP,
      `src/simsopt_jax/solve/permanent_magnet.py:919`) at nφ=64: first
      measure ndipoles/memory at that grid (unmeasured — grid is built
      per-φ-plane), then time if it fits; else record the blocker with the
      measured number.
- [x] `[charter]` P3.6 (2026-08-23 — draft written: Rung A pm-simple 5.2×, Rung B muse-64 2.9×, pm4stell BLOCKED on the k=201 fork with a pre-registered two-hypothesis adjudication) Charter draft for any family member clearing ≥1.10×
      warm; the nφ=64 rung lands as a `_scale_configuration()` addition in
      the relevant parity cases so the charter has a frozen scale to cite.

### Phase 4 — Marginal quartet (kill-fast probes only)

- [x] `[probe]` P4.1 (2026-08-23 — KILLED: 0.30×/0.33×, kill rule fired; policy+quadrature asymmetries disclosed and insufficient to flip the verdict) `stage_two_optimization` + `_planar_coils`: warm fused
      GPU (already the default lane) vs swept-OMP native at native_default,
      N=3 interleaved pairs; disclose the `serial_solve_jax` bounded-log
      overhead (`src/simsopt_jax/solve/serial.py:629-637`); kill at <1.0×
      warm.
- [x] `[probe]` P4.2 (2026-08-23 — NOT killed: ~1.6× GPU with a better endpoint objective; chartered as a rung of the stochastic campaign draft) `coil_forces`: same probe shape (0.92M pairs + 202.5k
      force terms per eval); kill at <1.0× warm.
- [x] `[probe]` P4.3 (2026-08-23 — KILL: 0.57× matched solve window; crossover prediction held at n_free=497) `wireframe_rcls_with_ports`: time the actual device
      solve (QR 497×254 + lstsq 1,521×243,
      `src/simsopt_jax_adapters/solve/wireframe.py:150,172-176`) vs the
      native RCLS at the shipped system; one probe, then close the row with
      the measured number.
- [x] `[doc]` P4.4 (2026-08-23 — recorded; scoreboard amendments per P6.2) Record outcomes in §Probe outcomes; propose `cpu`-row
      mechanism-text updates where a probe closes a question (amendment
      procedure; the rows stay `cpu`/`unmeasured` unless a certified receipt
      later moves them).

### Phase 5 — Nested-LS outer track (tracking only; fenced)

- [x] `[doc]` P5.1 (2026-08-23 status: FD-0 ran RED under Amendment 2
      (`fail_closed_reason=fd_rel_error_unmet`, 3/11 directions); Amendment 3
      descent ladder committed at `4f0b8bdbd`; canonical FD-0 rerun, B3, B37
      all pending, owned by the concurrent session) Track under the live charter (owned by the concurrent
      session): canonical FD-0 rerun → B3 → B37; no file in the fence list
      is touched by this plan.
      **Addendum (2026-08-23, later):** the canonical FD-0 rerun landed
      GREEN — 11/11 directions under the Amendment-3 descent ladder
      (`484b3fc26`); the owning session's native OMP sweep for B3 is in
      flight. B3 → B37 remain.
- [ ] `[doc]` P5.2 If B37 closes green, add a follow-up entry here for the
      F4-style example-promotion charter — noting the instrument-vs-example
      configuration gap (`surface_vessel_weight`, host driver) promotion
      must bridge.

### Phase 6 — Bookkeeping

- [x] `[doc]` P6.1 (2026-08-23) Fill §Probe outcomes for every probed family; every row
      carries scale, warm/cold, ratio vs swept-native, physics-check result,
      and artifact path.
- [x] `[doc]` P6.2 (2026-08-23 — ten-row amendment pass, drift test green) Amend `docs/jax_example_device_assignment.md` only where
      §Probe outcomes justifies a mechanism-text update, per its procedure;
      `gpu` moves wait for certified receipts from follow-up charters.
- [x] `[doc]` P6.3 (2026-08-23 — backlog-probe-campaign-2026-08-23.md + OpenMemory 94077643) Update session memory records (auto-memory + OpenMemory)
      with probe outcomes; move this plan's Status forward.

## Probe outcomes

All rows diagnostic-not-certifying; ratio = GPU vs swept-OMP native optimum,
warm windows matched per the probe's disclosures; artifacts under
`docs/receipts/evidence/` (2026-08-23 suffix) plus the shared leg ledger.

| Family | Scale | Lane ratio (warm) | Cold | Physics check | Artifact |
| --- | --- | --- | --- | --- | --- |
| rcls-with-ports | shipped | **0.57× (kill)** solve-window; 1.08× whole-window (sub-bar) | GPU 0.34 s | objectives rel 1.4e-14 | `marginal_rcls_*` |
| pm-simple | native 16×16, K=500 | **5.2× WIN** (0.0306 s vs omp32 0.1586 s; omp48 collapses 6.07 s) | GPU 0.10–0.46 s | bitwise, 0 ULP | `pm_simple16_*` |
| MUSE | shipped nφ=16 | **0.64×** (4.71 s vs omp32 3.01 s) — the row's 4.05× is stale (post-cond-skip) | GPU 5.6–6.5 s | bitwise, 0 ULP | `muse_shipped_*` |
| MUSE | nφ=64 "real run" | **2.9× WIN** (7.32 s vs omp32 21.22 s); device mem 10.5 GiB | GPU 8.3–8.8 s (beats all native) | bitwise, 0 ULP | `muse64_*` |
| PM4Stell | nφ=64 | 3.0× speed (9.51 s vs omp32 28.7 s) — **BLOCKED: greedy forks at k=201** (inputs digest-identical, lanes internally stable, agree ≤200) | GPU 10.5–11.0 s | **DIVERGES** from k=201; fork dumps archived | `pm4stell64_*` incl. `_fork_k201_*` |
| GSCO modular | reference 96×100/20k | **5.2× WIN** (warm median 5.27 s vs best omp32 leg 27.4 s; omp48 154 s) | GPU 5.4–6.1 s | bitwise, 0 ULP (19,200 seg) | `gsco_modular_reference_*` |
| GSCO sector-saddle | reference | **4.4× WIN** (warm median 8.08 s vs best omp32 leg 35.2 s; omp48 233 s) | GPU 8.3 s | bitwise, 0 ULP | `gsco_sector_saddle_reference_*` |
| GSCO both siblings | shipped 48×50/2k | **~1.6× GPU** (warm median 0.408 s vs omp32 0.669 s) — direction REVERSED vs the sealed 2026-08-16 receipt (0.79–0.89×); genuine conflict, adjudication chartered | GPU 0.51–1.56 s | bitwise, 0 ULP | `gsco_*_shipped_*` |
| P2.3 per-iteration | both | GPU 0.20→0.26 ms/it (2k→20k, near-flat); native 0.335→1.385 ms/it | — | — | same artifacts |
| stochastic | shipped, mc10 / mc400, nit=400 both lanes | **1.35× / 1.24× WIN** (22.15 / 24.62 s vs omp16 29.99 / 30.53 s) | GPU 157–214 s (XLA compile — heavy) | endpoints within `native_workflow` bucket; samples bitwise-shared | `stoch_*` |
| stochastic tiles | shipped, tile ∈ {4,8,16} | 24.94–25.21 s vs 24.62 s untiled — **lever neutral-to-negative** | — | — | `stoch_jaxgpu_mc400_tile*` |
| stage-two / planar | shipped-vs-shipped | **0.30× / 0.33× — KILL** (34.5 / 15.3 s vs 10.4 / 5.0 s; policy + 1.33× quadrature asymmetry disclosed, anti-GPU direction) | GPU 38.6–76.2 s | endpoint objectives within family tolerance | `quartet_stage_two_*`, `quartet_planar_*` |
| coil-forces | shipped-vs-shipped, full two-stage | **~1.6×** (24.5 s vs best native 40.0 s of noisy 40–82 s legs); GPU endpoint objective BETTER (2.77e-5 vs 2.9e-5) | GPU 26.7–57.5 s | report-only objective compare | `quartet_coil_forces_*` |
| QA relax-split | nφ=64 | **BLOCKED**: GPU fail-closes on the alpha bound (native 2(1−1e−5)/ATA_scale > JAX 2/λ_max(H) by ~1e-4); native ~32.4 s (OMP-insensitive) | — | n/a | `qa64_*` |

## Validation Plan

- [x] (2026-08-23: 11 passed after the ten-row amendment) `pytest tests/test_jax_example_device_assignment.py` green after any
      scoreboard edit.
- [x] (2026-08-23: 101 tests across six files, one per process) Per-family focused tests green, one file per process (JAX x64 rule):
      stochastic tile-parity test (P1.2), pm_simple mirror scale tests
      (P3.1), probe-conventions unit test (P0.1).
- [x] `bash scripts/lint.sh check` AND `format` clean on every file this
      plan touches (the pinned gate; unpinned `uvx ruff` is the documented
      drift trap — a format-only regression slipped past `check` alone once
      during review).
- [x] (2026-08-23: identity + ledger present in every published artifact; executed order in `docs/receipts/evidence/probe_leg_ledger.jsonl`) Every probe artifact contains per-leg identity JSON (commit, observed
      OMP, jax platform/x64, device list, nvidia-smi processes) and the
      interleave order; probes without them are invalid and rerun.
- [x] (2026-08-23: GSCO bitwise at both scales; GPMO bitwise except the pm4stell k=201 fork, recorded and blocked; stochastic within bucket) Physics checks per family: GSCO endpoints bitwise (proven 0 ULP at
      shipped scale); GPMO endpoints bitwise expected (proven for baseline
      at 16×16; ArbVec at nφ=64 has never been compared — any nonzero diff
      is recorded and gated at `native_workflow`); stochastic and
      stage-two-family endpoints within the `native_workflow` tolerance
      bucket (`src/simsopt_jax/parity_tolerances.py`).
- [x] (2026-08-23: fence diff empty at every commit) No diff under the Phase-5 fence paths (`git diff --stat` inspected
      before every commit).

## Risks and Mitigations

- Risk: concurrent session commits race this worktree (observed twice on
  2026-08-23). Mitigation: fence list; re-verify anchors and rebase before
  each commit; commits scoped via commit-only-work; never bare `git stash`.
- Risk: reference-scale GSCO aggregate structure (one flat 20,000-it stage)
  amortizes launches differently than multistep's staged loop — the 3.5×
  may not transfer. Mitigation: P2.3 measures per-iteration time at both
  scales; kill rule at <1.0× warm.
- Risk: stochastic tiled-vmap changes the sample-sum reduction order, so the
  tile path may miss bit-identity with the scan oracle. Mitigation: gate at
  the `native_workflow` bucket, record max diff, keep scan as oracle and
  fallback; any charter freezes the tile size.
- Risk: PM nφ=64 XLA peak exceeds the analytic estimate (estimates are
  array-sum, not XLA-schedule, accounting). Mitigation: run P3.4 with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` first at K=100 to observe peak, then
  full budget; PM4Stell (6.8 GB estimate) after MUSE (3.3 GB).
- Risk: box contention poisons native denominators (native OpenMP collapses
  under load, 50–228× observed in the winnable-six campaign). Mitigation:
  interleaved A/B always; loadavg recorded per leg; discard legs with
  foreign GPU processes in nvidia-smi.
- Risk: probe numbers get quoted as claims. Mitigation: every probe artifact
  and §Probe outcomes row carries `diagnostic-not-certifying`; charters are
  the only claim path.

## Completion Criteria

- [x] Phases 0–4 and 6 tasks each checked or closed with a named blocker
      recorded in place (Phase 5 is tracking-only and closes with a status
      note).
- [x] §Probe outcomes has a row for every probed family (or a named blocker
      where a probe could not run).
- [x] Charter drafts exist for every family that cleared 1.10× warm; none
      exists for a family that didn't.
- [x] Validation Plan checkboxes all green at final commit.

## Open Questions

- Stochastic: does the tiled-vmap lever's speedup survive the fused driver's
  per-iteration overhead at shipped scale, and which tile size wins? (P1.3
  decides.)
- PM QA: ndipoles and device memory at nφ=64 for the
  `geo_setup_between_toroidal_surfaces` grid. (P3.5 measures.)
- GSCO siblings: does the flat 20,000-iteration structure amortize like
  multistep's staged loop? (P2.3 measures.)
- RCLS: where does the null-space ECLS shape (QR 497×254 + lstsq 1,521×243)
  actually sit relative to the host-local dense-solve crossover? (P4.3
  measures.)
