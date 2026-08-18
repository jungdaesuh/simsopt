# Finite-build Stage-II successor campaign — terminal receipt

**Verdict: `WIN`** (2026-08-18, preregistered five-pair rule). At matched,
oracle-verified physics quality, the fused JAX GPU lane beats the fastest
qualifying native SIMSOPT/simsoptpp CPU lane on both preregistered timers:

| Timer | Native (median) | GPU (median) | Pair ratios | Median ratio | Gate |
| --- | --- | --- | --- | --- | --- |
| `warm_solve_seconds` | 45.23 s | 3.353 s | 13.77 / 13.96 / 13.01 / 13.45 / 13.58 | **13.58×** | ≥ 1.10 ✔, every pair > 1.00 ✔ |
| warm persistent-cache `process_wall_seconds` | 50.1 s | 16.11 s | 3.24 / 3.19 / 3.11 / 2.90 / 2.97 | **3.11×** | ≥ 1.10 ✔, every pair > 1.00 ✔ |

**The claim is a warm same-process and warm persistent-cache repeated-workload
win only.** The separately measured fresh-empty-cache run is a **bounded
negative for cold-start**: warm ratios reproduce (median 13.82×) but a fresh
process pays the ~42 s XLA compile, so its process wall loses (native
52.0 s vs GPU 59.1 s, median ratio **0.88×**, `CLOSED_BOUNDED_NEGATIVE`
under the same five-pair rule). No cold-start claim is made. A cold GPU
process on this workload is ~0.9× native; the win exists for repeated
workloads and persistent-cache launches.

Charter: `docs/jax_gpu_finitebuild_native_speed_successor_plan.md` (the new
preregistration the predecessor's close-path receipt called for). Predecessor
close: `docs/receipts/stage_two_finitebuild_native_gpu.md`
(`CLOSED_BOUNDED_NEGATIVE`, protocol-shaped — superseded as a speed verdict
by this campaign's symmetric protocol, not contradicted: its gate and this
gate are different frozen contracts).

## What was measured, exactly

Both lanes time "solve until the first accepted iterate at
native-equivalent quality" under one frozen v4 contract
(`stage-two-finitebuild-quality-contract-v4-successor`, sha256
`afef656be7de7fad…`, target objective `5.4666e-07` = 1.001× the converged
fp64 reference, reference budget 398):

- **Native lane** (denominator): the fastest fully-qualifying configuration
  from a 24-configuration matrix — OMP=2, L-BFGS-B history 400 — stopping
  via its per-iteration callback at the frozen rung, re-crossing per
  repetition at nit 374–401. Its callback cost is charged to the native
  lane (≤1e-5 relative, four orders under the gate). The shipped-default
  native configuration is slower (~71 s); the denominator is the swept
  optimum, per the fair-native rule.
- **GPU lane** (numerator): the fused on-device L-BFGS loop (history 10) at
  fixed budget k\*=500 — its exact crossing iteration, found by bisection
  and proved minimal (probes at 499 and 500) — with **every pair's endpoint
  verified bitwise against the frozen crossing solution**
  (`8102d99a33c6ae06…`). Endpoint quality was gated through the **native
  oracle** (`native-endpoint-eval` at the published solution); equivalence
  was never mediated by the JAX lane's own evaluator. The gradient clause
  is the v4 window-median cap (2.3× the 21-iterate window median), which
  both lanes' published endpoints satisfy. One disclosed deviation from the
  frozen charter: the h10 bisection ran **11** probe legs
  (400/450/475/487/493/496/498/499/500/600/800) against the charter's
  "≤ 10 probe solves per history" — the charter's arithmetic counted the
  10 halvings of [0, 800] but omitted the cap-seed leg at b = 800, which
  the reducer requires. The bound governs instrument cost only; probes are
  untimed and unranked, so the extra leg cannot flatter either lane.

Five interleaved pairs per cache policy, alternating order, pinned CPU
affinity and OpenMP environments, serialized GPU, box-idle gates on every
timed leg, fp64 child-observed conformance, gate-source conformance
(physics pins fail-closed; benchmark/plan drift disclosed — all
`identical: true` here).

## Evidence chain (all clean-tree at commit `66003ee45`)

| Phase | Run | `manifest.json` sha256 | Result |
| --- | --- | --- | --- |
| gate | `20260818T013720Z-gate-3399135` | (contract `afef656be7de7fad…`) | `GATE_FROZEN` — v4; freeze audit passed (window median `1.914e-06`, anchor `1.140e-06`, ratio 1.678) |
| baseline | `20260818T013827Z-baseline-3400714` | `1894150a8c9a7796…` | `IDENTITY_OK` |
| kernel canary | `20260818T013842Z-kernel-canary-3401563` | `a48c7d0db7dfb3ba…` | `PROCEED` **12.76×** (GPU 5.24 ms vs best native 66.9 ms @ OMP=2) |
| native matrix | `20260818T014134Z-native-matrix-3408919` | `fb7c36300d516f74…` | `NATIVE_SELECTED` — omp2-h400, 45.108 s, nit 401; 7 of 24 eligible (within 1% of the charter's pre-registered ≈44.8 s archived-data estimate) |
| jax sweep | `20260818T031456Z-jax-sweep-3598504` | `5397180e39289ce1…` | `JAX_SELECTED` — h10, k\*=500, warm median 3.250 s; h20/h40 never reach the rung within b ≤ 800 |
| selection | `successor-selection/selection.json` | `f457b0adf65409e4…` | frozen |
| **final pairs (primed)** | `20260818T195452Z-final-pairs-primed-1545361` | `86faf680eb0ecc01…` | **`WIN`** (the verdict) |
| final pairs (fresh) | `20260818T202021Z-final-pairs-fresh-1676537` | `607b550d8a9183eb…` | `CLOSED_BOUNDED_NEGATIVE` (cold-start; reported, no claim) |

## Where the runs executed — and what licenses the transfer (disclosure)

The final pairs ran in a git worktree pinned at **`66003ee45`** — the exact
commit every selection-chain run validates against — because the S5
production commit (`ead83eaef`) legitimately edited a **physics-pinned
source** (the parity case, `parity_case_sha256`), and the gate-source
conformance clause fails closed on any pin movement by design. Three facts
make the measurement transferable to the shipped code:

1. The pairs measure harness legs, which never import the production
   module; the harness, plan, and charter hashes are `identical: true`
   between the gate and every pair row.
2. The objective module (`objective_module_sha256`) — the physics — is
   byte-identical at `66003ee45` and the S5 tree.
3. **The shipped production module reproduces the frozen crossing solution
   bitwise at the S5 tree** (nit 500, solution sha `8102d99a33c6ae06…`):
   `evidence/…/module-reproduction/module_reproduction.json`, which binds
   the S5 sources it executed. (Two self-referential artifacts of its
   write order, disclosed there: it binds the pre-regeneration manifest
   hash, and it cannot bind its own file.)

## Production deliverable

`src/simsopt_jax/examples/stage_two_finitebuild.py` (commit `ead83eaef`):
the internal workflow both JAX callers route through — frozen history 10
(no configuration knob), `dispatch.minimize` with typed options, no host
observation inside the solve (strict-transfer suite: positive
`final_result` control, zero `advance`/`callback`/`unclassified`, transfer
ledger identical across step budgets). Execution-source manifest 615 → 616
with both count twins.

## Tracked evidence

`docs/receipts/evidence/stage_two_finitebuild_native_gpu_successor/`
(~1.5 MB, 142 files, every copy byte-identity-verified against its
original): both final-pairs runs **complete and directly re-validatable**
(all rows + launch rows + manifest + stored gate and selection, preserving
the run layout — `benchmarks/stage_two_finitebuild_native_gpu.py validate`
on the tracked copies recomputes `WIN` and `CLOSED_BOUNDED_NEGATIVE` with
zero gate failures), the five selection-chain manifests,
the frozen quality contract, the frozen selection, the
protocol-determinism probe rows (cold/primed/warm bitwise identity), and
the module-reproduction row. Raw run directories remain
**[host-local]** under `.artifacts/` (main tree) and the
`simsopt-finitebuild-pairs` worktree; each is integrity-bound by its
tracked manifest's per-row sha256 map.

## Lineage of honest negatives this result rests on

- Predecessor campaign: closed bounded-negative on a landing-clause/protocol
  asymmetry rather than minting a win from an unfair gate.
- fp64 taint: the native lane's transitive JAX ran float32 until
  2026-08-17; caught by the oracle cross-check, fixed (`6bce010d0`), all
  tainted evidence voided and regenerated.
- Budget-parity and gradient-window amendments: each dated, each
  pre-evidence, each with its empirical basis archived.
- Cold-start: measured, lost, reported — the claim is exactly as wide as
  the evidence.
