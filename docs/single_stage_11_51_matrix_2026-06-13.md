# Single-stage 11-vs-51 production test matrix

- Matrix id: `single_stage_11_51_2026-06-13`
- Source SHA: `0752b18f12255b185a9f6c48cb22fd638149ebec`
- Seed: `benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json`
- Budgets: outer L-BFGS 1500, Boozer BFGS 1500, Boozer Newton 50, polish `run`
- Inner Boozer least-squares: `quasi-newton` (both lanes; LM/lm-minpack dropped)

## Formulation/backend coupling

- `11` reduced (coils only, surface solved each iteration by the inner Boozer solve) ⇒ `scipy-jax` (host SciPy L-BFGS-B).
- `51` 51-dim coils+surface vector (full-graph `JF.x` DOF map) ⇒ `scipy-jax-fullgraph` (host SciPy L-BFGS-B over the full DOF vector; outer gradient on the coil block only, surface re-solved each iteration by the inner Boozer solve, not a residual penalty).
- Both lanes are host-driven (one value/grad eval compiled per step); the monolithic `ondevice` lane is removed (it OOMs at production resolution).
- The inner Boozer solve runs on-device (JAX) on both lanes; the `scipy-jax-fullgraph` cells force `--boozer-optimizer-backend ondevice` (its CPU default is otherwise `scipy`, rejected under `jax_cpu_parity`).
- The cpp/CPU reference is always 51; the 11-dim `scipy-jax` lane is compared against a dim-mismatched 51 reference, while the 51-dim `scipy-jax-fullgraph` lane is dim-matched.

## Cells

| id | dim | backend | inner-boozer | platform | tier | status | ref dim match | runnable now |
|---|---|---|---|---|---|---|---|---|
| `ss_11_scipyjax_cpu_mpol2` | 11 | scipy-jax | default | cpu | mpol2 | core | NO (51 vs 11) | yes |
| `ss_11_scipyjax_cpu_mpol10` | 11 | scipy-jax | default | cpu | mpol10 | core | NO (51 vs 11) | no (donor) |
| `ss_11_scipyjax_gpu_mpol2` | 11 | scipy-jax | default | gpu | mpol2 | core | NO (51 vs 11) | yes |
| `ss_11_scipyjax_gpu_mpol10` | 11 | scipy-jax | default | gpu | mpol10 | core | NO (51 vs 11) | no (donor) |
| `ss_51_scipyjaxfullgraph_cpu_mpol2` | 51 | scipy-jax-fullgraph | ondevice | cpu | mpol2 | core | yes | yes |
| `ss_51_scipyjaxfullgraph_cpu_mpol10` | 51 | scipy-jax-fullgraph | ondevice | cpu | mpol10 | core | yes | no (donor) |
| `ss_51_scipyjaxfullgraph_gpu_mpol2` | 51 | scipy-jax-fullgraph | ondevice | gpu | mpol2 | core | yes | yes |
| `ss_51_scipyjaxfullgraph_gpu_mpol10` | 51 | scipy-jax-fullgraph | ondevice | gpu | mpol10 | core | yes | no (donor) |

Total cells: 8 (8 core, 0 extended).

## Notes

- Both lanes run the outer loop on the host (SciPy L-BFGS-B); only one value/grad eval is compiled per step, so neither builds the monolithic on-device graph that OOMs. 11=scipy-jax (reduced), 51=scipy-jax-fullgraph (full JF.x). The ondevice lane is removed.
- Formulation dim is hard-coupled to the optimizer backend; no flag decouples them.
- Every run also yields the cpp/CPU reference at 51 full-space; there is no coil-only native reference in this harness, so the 11-dim scipy-jax lane has a dim-mismatched 51 reference (performance/feasibility), while the 51-dim scipy-jax-fullgraph lane is dim-matched and replay-capable (exact same-candidate bit-parity).
- Inner Boozer least-squares is quasi-newton for both lanes; LM/lm-minpack and optax-lbfgs/optimistix-lbfgs are intentionally excluded.
- scipy-jax-fullgraph cells force --boozer-optimizer-backend ondevice (via PROD_BOOZER_OPTIMIZER_BACKEND): the fullgraph inner Boozer otherwise defaults to scipy on CPU, which jax_cpu_parity rejects at boozer_surface.py:5659. The GPU auto-default already supplies ondevice. Confirmed by the b5f97fdf9 gate smoke.
- GATE: scipy-jax-fullgraph (51) has no passing artifact yet (its only completed run pre-fix failed; the b5f97fdf9 smoke cleared the free_x bug and ran 5 outer iterations on GPU but did not reach passed=true -- CPU needed the boozer-backend fix above; GPU is compile-bound). A mpol=2 smoke must show rc=0/passed=true at the source SHA before any mpol=10 fullgraph cell is submitted.

## Tier detail

- `mpol2`: mpol=2 ntor=2 nphi=31 ntheta=16, binds_budget=False, runnable_now=True. smoke resolution: loose tolerances (gtol 1e-2), budgets do not bind, optimizers stop in ~5-219 steps; cheap diagnostics tier. The scipy-jax-fullgraph mpol2 cells double as the Phase 0 GATE smoke (inspect their result JSON for passed=true).
- `mpol10`: mpol=10 ntor=10 nphi=64 ntheta=32, binds_budget=True, runnable_now=False. production resolution: tolerances tighten (gtol 1e-7), 1500/50 budgets bind; requires --warm-start-run-dir from the continuation donor build (2->4->6->8->10 ladder via --intermediate-rungs). Both host-SciPy lanes compile only one value/grad eval per step, so node memory is bounded by a single eval rather than the whole optimization (no monolithic jit).
