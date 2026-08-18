# Finite-build Stage-II native-vs-GPU speed route — terminal receipt

> **Superseded as a speed verdict (2026-08-18):** the successor campaign this
> receipt's close path called for was chartered
> (`docs/jax_gpu_finitebuild_native_speed_successor_plan.md`) and measured a
> **`WIN`** under a symmetric first-crossing protocol — for warm and
> persistent-cache repeated workloads; cold start was measured and lost —
> `docs/receipts/stage_two_finitebuild_native_gpu_successor.md`. The verdict
> below stands unchanged under its own frozen contract; the two gates are
> different frozen contracts, so this receipt is superseded, not contradicted.

**Verdict: `CLOSED_BOUNDED_NEGATIVE`** (2026-08-17, preregistered Step-3 kill
criterion, final at budget parity). The fused GPU lane reaches the frozen
objective rung — `h10-b560` converges 0.524% *below* the target with every
quality cap and geometry band clean — but the endpoint it publishes fails
the gradient infinity-norm landing clause (ratio 1.98 vs the 1.05 cap) at
every preregistered budget up to native parity (`b ≤ 800`). No JAX history
reached the frozen endpoint contract, selection never froze, and no
native-vs-GPU solve-time comparison was produced. **This receipt makes no
speed claim in either direction**, and per the plan these changes are not
extended to stochastic Stage II.

Campaign plan and amendment log:
`docs/jax_gpu_finitebuild_native_speed_implementation_plan.md` (all
amendments dated 2026-08-17, each made before the evidence it governs).
Commits: harness + contract `aa661826d`; fp64-taint fix `6bce010d0`;
budget-parity amendment `8435ad814` (the terminal evidence commit — every
regenerated run below validates `clean-tree` at `8435ad814` or, for the
pre-amendment runs, at `6bce010d0`).

## Terminal evidence chain (all fp64, all clean-tree)

| Phase | Run directory (`.artifacts/stage_two_finitebuild_native_gpu/`) | `manifest.json` sha256 | Result |
| --- | --- | --- | --- |
| gate | `20260817T191738Z-gate-2450592` | `8f6ccfd583c9c213…` | `GATE_FROZEN` — contract sha256 `872834d86a16fef6…`, reference budget 398, unscaled target `5.455605e-07` (`1.001×` converged reference), objective scale `1e-4` |
| baseline | `20260817T191843Z-baseline-2452184` | `d8a23455acba337d…` | `IDENTITY_OK` — native/JAX value-gradient-diagnostic identity at initial + two perturbed states |
| kernel canary | `20260817T191858Z-kernel-canary-2452727` | `f5669f75848a45b3…` | `PROCEED` at **13.03×** — GPU warm value/grad median 5.166 ms vs best native 67.33 ms at OMP=2 (native degrades with thread count, best at OMP=2: 94.6/97.7/115.2/140.3/102.1 ms at OMP 4/8/16/32/48) |
| native matrix | `20260817T192150Z-native-matrix-2459973` | `5509b8609c8bc8bf…` | `NATIVE_SELECTED` — omp2-h10, median fresh-process solve **81.657 s** to the rung, median nit **736** of the 800 cap; **1 of 24 configurations eligible**; shipped-default disclosure lane: 72.757 s solve, nit 409, endpoint `5.4551e-07` — **does not qualify**: it fails the gradient landing clause at ratio 1.64 vs 1.05, reproducing on the native side the exact failure mode that closes the GPU route |
| jax sweep (5-rung, pre-amendment) | `20260817T205932Z-jax-sweep-2666978` | `cc23b02e5cd2e667…` | `CLOSED_BOUNDED_NEGATIVE` under the original ladder (verdict archived verbatim in the budget-parity amendment; post-amendment revalidation returns `NOT_PRODUCED` by construction) |
| jax sweep (7-rung, terminal) | `20260817T211803Z-jax-sweep-2715269` | `93c022a61c7db150…` | `CLOSED_BOUNDED_NEGATIVE` — "no JAX history reached the frozen endpoint contract"; kill final at `b ≤ 800` |

Every GPU endpoint above was gated through its **native re-evaluation**
(`native-endpoint-eval` oracle leg at the published solution vector);
equivalence was never mediated by the JAX lane's own evaluator.

Validation additionally binds the frozen gate to every consuming run's own
sources: the physics pins (objective module, parity case) are enforced
**fail-closed** — a consumer whose physics differs from the gate's is
`NOT_PRODUCED`, never a verdict — and both pass here (the physics sources
are byte-identical at `6bce010d0` and `8435ad814`). The harness and plan
pins moved between the gate commit and the terminal sweep through the dated
budget-parity amendment, and the validator publishes that drift verbatim
(`gate_source_drift`: benchmark `ad2cde3118…` → `f608cb8340…`, plan
`6106d772bb…` → `0b1c6d954a…`, identical `false`) instead of silently
accepting or rejecting it.

## Tracked evidence

Byte-identical copies of every file this receipt's numbers derive from are
committed under `docs/receipts/evidence/stage_two_finitebuild_native_gpu/`
(260 KB: the six run manifests, the frozen quality contract, the selected
native configuration's three timed rows plus the shipped-default disclosure
row, and the decisive terminal-sweep rows `h10-b560`/`h10-b800` with their
native oracle re-evaluations). Full hashes — the table above abbreviates
these:

```
8f6ccfd583c9c213ca634b71ebab97d37dd8452d1034acb46f0ee2577b768ed5  gate manifest.json
872834d86a16fef6b3d403bcdc5e31f8e3d2a86b744ee5a90b56e039e9102039  gate quality_contract.json
d8a23455acba337d6b5ffb3f2f230f855740f24790c7c0e0fd914b4f8f4089a2  baseline manifest.json
f5669f75848a45b30f154d28b934cb81d8a47e87b668fc00c23143f6a8d8be64  kernel-canary manifest.json
5509b8609c8bc8bfa6a8e30d93c251c3e33e8dd4efaadc8b81d1810e3821dda2  native-matrix manifest.json
cc23b02e5cd2e66705918b485251851327b7130dc9ab5fff61d7bfe6693e423b  jax-sweep (5-rung) manifest.json
93c022a61c7db150367beb73268560eec27b797bcfaed9e1dfb3e655c0ac3f61  jax-sweep (terminal) manifest.json
```

The remaining raw rows (~54 MB) are **[host-local]** under `.artifacts/`
on this workstation, per this repository's evidence-provenance convention;
each one's integrity is bound by its run's tracked `manifest.json`, which
enumerates every expected row file with its sha256.

## The terminal sweep table (oracle-evaluated endpoints vs the frozen gate)

Unscaled objective vs target `5.455605e-07`; gradient ratio vs the truncated
anchor's `|g|∞ = 1.349e-06` under the 1.05 cap:

| Config | objective | vs target | `|g|∞` ratio | nit | solver status |
| --- | --- | --- | --- | --- | --- |
| h10-b400 | `5.719e-07` | +4.83% | 3.54 | 400 | 1 (budget) |
| **h10-b560** | **`5.427e-07`** | **−0.52%** | **1.98** | **545** | **0 (converged)** |
| h10-b800 | `5.427e-07` | −0.52% | 1.98 | 545 | 0 (identical — solver terminates at 545) |
| h20-b800 | `9.655e-07` | +76.98% | 78.3 | 556 | 0 (stalled above rung) |
| h40-b800 | `8.826e-07` | +61.78% | 69.9 | 349 | 0 (stalled above rung) |

(Fuller ladders in the run's rows; h20/h40 are genuine per-history negatives
— they plateau far above the rung on their own convergence tolerances.)

## What the bound is — and is not

- **The binding clause is the gradient landing condition**, disclosed in the
  budget-parity amendment *before* this evidence existed: the clause anchors
  to the reference run's first rung-crossing iterate, `|g|∞` oscillates along
  trajectories, and the reference's **own converged endpoint fails the
  clause** (ratio 2.08). The GPU terminal endpoint at 1.98 lands *closer to
  the anchor* than the reference's own final iterate — but the frozen
  contract compares landings, and the kill was preregistered as final at
  budget parity. Amending the clause after watching it become the sole
  binding clause would be post-hoc, so the verdict stands.
- **The bound is protocol-shaped**: the native lane publishes its *first
  rung-crossing iterate* (stopping callback), while the fused GPU lane
  publishes only its terminal iterate — its own first-crossing iterate near
  nit ~545 was never captured, because the on-device loop records no
  trajectory. The native lane corroborates the severity of the clause: 23
  of 24 matrix configurations fail the contract — 16 never reach the rung
  in any repetition (of the 49 matrix legs that never reached the rung, 22
  exhausted the 800-iteration cap and 27 stopped below it, at nit 93–693,
  on the solver's own criteria; the other 24 legs were callback-stopped at
  the rung), 6 (the h400 history at every thread count) fail only the
  gradient landing clause, and 1 fails both —
  and the shipped-default disclosure lane crosses the rung with every cap
  and band clean yet fails only the landing clause (1.64 vs 1.05) *despite
  being callback-stopped at its first crossing iterate*. First-crossing
  capture is therefore necessary but not sufficient: of the 24 native legs
  the callback stopped at their first rung-crossing iterate, only 6 landed
  — and three of those are lone repetitions of configurations whose sibling
  repetitions missed by more than 2× (omp16-h10 at 0.893/2.290/cap-exhausted;
  omp16-h400 at 0.947/1.721/1.339; omp32-h400 at 0.880/1.627/1.814), the
  cleanest demonstration that clearing the clause is a landing, not a
  descent. Only omp2-h10 landed in all three repetitions (0.955 each) — the
  sole eligible configuration — while the shipped default (1.638) missed
  and no h400 configuration was eligible. The GPU lane never got even the
  necessary condition.
- **No speed claim exists in either direction.** The kernel canary's 13.03×
  is a value/gradient kernel measurement under its own protocol, not a solve
  claim; the native 81.657 s denominator was measured, but no GPU
  configuration qualified to stand against it. The device-assignment row for
  `native-stage-two-optimization-finitebuild` therefore stays `unmeasured`.

## Disclosed defect: the fp32-tainted first pass (all void)

The first full pass (gate `83118aec…`, baseline, kernel canary 12.51×,
native matrix omp2-h10 at 49.24 s, and a jax sweep killed by the oracle
cross-check) ran with the native lane's transitively imported JAX in
**float32** — the env scrub dropped `JAX_ENABLE_X64` and nothing re-pinned
it. The oracle cross-check caught the resulting gradient fork (on the native
side; the GPU lane was right). Root-cause probes, verbatim outputs, and row
hashes: `docs/jax_gpu_finitebuild_fp64_taint_diagnostic.md`. Void run
directories are retained under `.artifacts/` as the defect's evidence:
`20260817T134021Z-gate-1637587`, `20260817T134228Z-baseline-1648737`,
`20260817T170754Z-kernel-canary-2170412`,
`20260817T171212Z-native-matrix-2179206`,
`20260817T183358Z-jax-sweep-2352781`.

## What would change the answer (successor charter, not chartered here)

A new preregistration — not an amendment — could make endpoint selection
symmetric: capture the fused GPU lane's first rung-crossing iterate on
device (a running argmin/first-crossing state in the loop carry), or
re-derive the landing clause against published terminal endpoints from
scratch. Either is a different contract and must bring its own gate
derivation — and the necessary-not-sufficient evidence above (first-crossing
capture landed only 6 of 24 native legs) means symmetric capture alone
carries no promise of a different verdict.
